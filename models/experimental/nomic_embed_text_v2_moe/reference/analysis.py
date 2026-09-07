# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generates MODEL_ANALYSIS.md: the Step 1 reference-model analysis for the TTNN port.

Covers the porting-skill Step 1 deliverables that are mechanical rather than prose: the
operator inventory, per-operator and per-module tensor shapes, parameter counts and memory,
the module hierarchy, and the model graph.

Run from the repo root:

    python -m models.experimental.nomic_embed_text_v2_moe.reference.analysis
    python -m models.experimental.nomic_embed_text_v2_moe.reference.analysis --dot out.dot

Operators are captured with TorchDispatchMode on a real forward rather than torch.fx.
symbolic_trace fails on this model twice over: torch.finfo() rejects a Proxy dtype in
build_extended_attention_mask, and the MoE expert loop branches on tensor values. Dispatch
capture records what actually executes, which is what the port has to reproduce anyway.

Synthetic weights by default, so this needs no network and no 1.8 GB download. Shapes,
operators and hierarchy do not depend on weight values.
"""

from __future__ import annotations

import argparse
import collections
import textwrap
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from models.experimental.nomic_embed_text_v2_moe.common import build_synthetic_model, random_input_ids
from models.experimental.nomic_embed_text_v2_moe.reference.configuration_nomic_moe import (
    NomicMoEConfig,
    load_vendored_config,
)

OUTPUT_PATH = Path(__file__).parent / "MODEL_ANALYSIS.md"

# Small but not degenerate: batch 2 with ragged padding exercises the mask path, and a
# sequence longer than one tile keeps the shapes representative.
SAMPLE_BATCH = 2
SAMPLE_SEQLEN = 16
SAMPLE_PAD_LENGTHS = [0, 5]

# The model's trained maximum, from sentence_bert_config.json via pipeline.MAX_SEQ_LENGTH.
MAX_SEQLEN = 512

BYTES_PER_FP32 = 4
BYTES_PER_BF16 = 2


class OperatorLog(TorchDispatchMode):
    """Records every dispatched aten op with a representative input/output shape."""

    def __init__(self):
        self.counts: collections.Counter = collections.Counter()
        self.shapes: dict[str, tuple[list, object]] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func)
        self.counts[name] += 1
        if name not in self.shapes:
            self.shapes[name] = (_tensor_shapes(args), _first_output_shape(out))
        return out


def _tensor_shapes(args, limit: int = 3) -> list[tuple[int, ...]]:
    return [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)][:limit]


def _first_output_shape(out):
    if isinstance(out, torch.Tensor):
        return tuple(out.shape)
    if isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor):
        return tuple(out[0].shape)
    return None


def capture_operators(model, input_ids, attention_mask) -> OperatorLog:
    log = OperatorLog()
    with torch.no_grad(), log:
        model(input_ids, attention_mask=attention_mask)
    return log


def _is_elided_repeat(name: str, keep_layers_up_to: int) -> bool:
    """True for encoder layers past the cutoff, which repeat the dense/MoE pair verbatim."""
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "encoder" and parts[1] == "layers" and parts[2].isdigit():
        return int(parts[2]) > keep_layers_up_to
    return False


def capture_module_shapes(model, input_ids, attention_mask, keep_layers_up_to: int = 1):
    """Per-module input and output shapes, via forward hooks, in execution order.

    Hooks every module but reports only layers 0 and 1; the remaining ten repeat that pair
    exactly, and listing all twelve buries the signal in 200 identical rows.
    """
    rows: list[tuple[str, str, str, str]] = []
    handles = []

    def make_hook(name: str, kind: str):
        def hook(_module, inputs, output):
            rows.append((name, kind, _fmt_shapes(_tensor_shapes(inputs)), _fmt_shape(_first_output_shape(output))))

        return hook

    for name, module in model.named_modules():
        if not name or isinstance(module, torch.nn.ModuleList):
            continue
        if _is_elided_repeat(name, keep_layers_up_to):
            continue
        handles.append(module.register_forward_hook(make_hook(name, type(module).__name__)))

    try:
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()

    return rows


def _fmt_shape(shape) -> str:
    return "-" if shape is None else str(list(shape))


def _fmt_shapes(shapes) -> str:
    return ", ".join(str(list(s)) for s in shapes) if shapes else "-"


def module_hierarchy(model, skip_repeats_above: int = 1) -> list[str]:
    """Indented module tree. Encoder layers above the index cutoff are elided as a repeat."""
    lines: list[str] = []

    def walk(module, prefix=""):
        for name, child in module.named_children():
            if name.isdigit() and int(name) > skip_repeats_above:
                continue
            lines.append(f"{prefix}{name}: {type(child).__name__}")
            walk(child, prefix + "  ")

    walk(model)
    return lines


def parameter_table(model) -> list[tuple[str, int, float, float]]:
    """Per-top-level-group parameter counts and memory, plus a total row."""
    groups: collections.Counter = collections.Counter()
    for name, param in model.named_parameters():
        if name.startswith("encoder.layers."):
            layer_idx = int(name.split(".")[2])
            config = model.config
            kind = "MoE block" if config.is_moe_layer(layer_idx) else "dense block"
            groups[f"encoder.layers.* ({kind})"] += param.numel()
        else:
            groups[name.rsplit(".", 1)[0] or name] += param.numel()

    rows = [(g, n, n * BYTES_PER_FP32 / 1e6, n * BYTES_PER_BF16 / 1e6) for g, n in groups.most_common()]
    total = sum(p.numel() for p in model.parameters())
    rows.append(("total", total, total * BYTES_PER_FP32 / 1e6, total * BYTES_PER_BF16 / 1e6))
    return rows


def mermaid_graph(config: NomicMoEConfig) -> list[str]:
    """Model graph as a mermaid flowchart. GitHub renders this inline; it stays diffable.

    Each block carries an explicit input node so both residual edges are visible: post-norm
    adds the block input to the branch output *before* the norm, and a diagram that showed only
    the branch edge would misrepresent the one structural fact the port most needs to get right.
    """
    hidden = config.hidden_size
    experts = config.num_experts
    ffn = config.intermediate_size
    # Labels avoid square brackets and the "->" arrow: both are mermaid parsing hazards inside
    # node text, and this diagram cannot be render-tested in CI. Shapes read "(B, S)" instead.
    return [
        "flowchart TD",
        f'    ids(["input_ids (B, S)"]) --> we["word_embeddings<br/>Embedding {config.vocab_size} x {hidden}"]',
        f'    tte["token_type_embeddings<br/>Embedding {config.type_vocab_size} x {hidden}"] --> add0(("+"))',
        "    we --> add0",
        f'    add0 --> embln["emb_ln<br/>LayerNorm eps {config.layer_norm_epsilon}"]',
        '    amask(["attention_mask (B, S)"]) --> extmask["additive mask<br/>(B, 1, 1, S)"]',
        "    embln --> x0",
        "",
        '    subgraph blk0["encoder.layers.0 (dense; even layers)"]',
        "        direction TB",
        '        x0(["x"])',
        f'        a0["attn<br/>Wqkv {hidden} to {config.qkv_dim}, three-major split<br/>'
        f'rotary NeoX, SDPA is_causal false<br/>out_proj {hidden} to {hidden}"]',
        '        r0(("+"))',
        '        n0["norm1"]',
        f'        m0["mlp<br/>fc1 {hidden} to {ffn}<br/>GELU exact erf<br/>fc2 {ffn} to {hidden}"]',
        '        r1(("+"))',
        '        n1["norm2"]',
        "        x0 --> a0",
        "        a0 --> r0",
        "        x0 --> r0",
        "        r0 --> n0",
        "        n0 --> m0",
        "        m0 --> r1",
        "        n0 --> r1",
        "        r1 --> n1",
        "    end",
        "",
        "    n1 --> x1",
        '    subgraph blk1["encoder.layers.1 (MoE; odd layers)"]',
        "        direction TB",
        '        x1(["x"])',
        '        a1["attn<br/>same as dense"]',
        '        r2(("+"))',
        '        n2["norm1"]',
        f'        rt["router.layer {hidden} to {experts}, no bias<br/>'
        f'softmax fp32 over all {experts}<br/>top-{config.moe_top_k}, NOT renormalized"]',
        f'        ex["experts.mlp<br/>w1 ({experts}*{ffn}, {hidden}) applied transposed<br/>'
        f'GELU, then w2 same shape as-is"]',
        '        wsum(("weighted<br/>sum"))',
        f'        eb["plus experts.bias ({hidden})<br/>once, after the sum"]',
        '        r3(("+"))',
        '        n3["norm2"]',
        "        x1 --> a1",
        "        a1 --> r2",
        "        x1 --> r2",
        "        r2 --> n2",
        "        n2 --> rt",
        "        n2 --> ex",
        "        rt --> wsum",
        "        ex --> wsum",
        "        wsum --> eb",
        "        eb --> r3",
        "        n2 --> r3",
        "        r3 --> n3",
        "    end",
        "",
        f'    n3 --> rep["blocks 2 to {config.num_hidden_layers - 1}<br/>alternating dense / MoE"]',
        f'    rep --> lhs["last_hidden_state (B, S, {hidden})"]',
        "    extmask --> a0",
        "    extmask --> a1",
        "",
        f'    lhs --> pool["mask-weighted mean pool<br/>(B, {hidden})"]',
        '    pool --> trunc["matryoshka truncate<br/>optional, feature axis"]',
        '    trunc --> l2["L2 normalize"]',
        '    l2 --> emb(["embedding (B, d)"])',
    ]


def dot_graph(config: NomicMoEConfig) -> str:
    """Module-level graph in DOT, for those who want a rendered PDF or PNG.

    Deliberately the module graph, not the autograd graph: make_dot on a 475M-parameter model
    yields thousands of nodes that nothing can read.
    """
    lines = [
        "digraph nomic_embed_text_v2_moe {",
        "  rankdir=TB;",
        '  node [shape=box, fontname="Helvetica", fontsize=10];',
        '  ids [label="input_ids [B, S]", shape=ellipse];',
        '  we [label="word_embeddings\\n[250048, 768]"];',
        '  tte [label="token_type_embeddings\\n[1, 768]"];',
        '  embln [label="emb_ln (LayerNorm)"];',
        "  ids -> we; we -> embln; tte -> embln;",
    ]
    previous = "embln"
    for layer_idx in range(config.num_hidden_layers):
        kind = "MoE" if config.is_moe_layer(layer_idx) else "dense"
        node = f"blk{layer_idx}"
        fill = "lightblue" if kind == "MoE" else "white"
        lines.append(
            f'  {node} [label="layers.{layer_idx}\\nattn + {kind} mlp\\npost-norm", style=filled, fillcolor={fill}];'
        )
        lines.append(f"  {previous} -> {node};")
        previous = node
    lines += [
        '  pool [label="mean pool -> truncate -> L2", shape=box, style=dashed];',
        f"  {previous} -> pool;",
        '  emb [label="embedding [B, d]", shape=ellipse];',
        "  pool -> emb;",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _para(text: str, width: int = 96) -> list[str]:
    """Wrap a prose paragraph so the generated markdown matches the width of the hand-written docs."""
    return textwrap.wrap(text, width=width)


def _md_table(header: list[str], rows: list[tuple]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def build_markdown(config: NomicMoEConfig, model, log: OperatorLog, module_rows, param_rows) -> str:
    total_params = sum(p.numel() for p in model.parameters())
    activation_bytes = SAMPLE_BATCH * SAMPLE_SEQLEN * config.hidden_size * BYTES_PER_FP32

    lines = [
        "# Reference Model Analysis",
        "",
        "Step 1 of the porting workflow: operators, shapes, parameters, module hierarchy and the",
        "model graph. Architecture facts and the TTNN operator mapping live in",
        "[`ARCHITECTURE.md`](ARCHITECTURE.md); this file is the mechanical inventory the port is",
        "built against.",
        "",
        "Generated by `reference/analysis.py`. Regenerate with:",
        "",
        "```bash",
        "python -m models.experimental.nomic_embed_text_v2_moe.reference.analysis",
        "```",
        "",
        f"Sample input: `input_ids [{SAMPLE_BATCH}, {SAMPLE_SEQLEN}]` with ragged padding "
        f"{SAMPLE_PAD_LENGTHS}, synthetic weights at the real dimensions.",
        "",
        "## 1. Model graph",
        "",
        "Module-level, not autograd-level: `make_dot` on 475M parameters yields thousands of",
        "unreadable nodes. `reference/analysis.py --dot out.dot` emits the same graph in DOT for",
        "rendering to PDF or PNG with `dot -Tpdf`.",
        "",
        "```mermaid",
    ]
    lines += mermaid_graph(config)
    lines += [
        "```",
        "",
        "## 2. Module hierarchy",
        "",
        f"Encoder layers 2 to {config.num_hidden_layers - 1} repeat the layer 0 (dense) and layer 1",
        "(MoE) pattern and are elided.",
        "",
        "```",
    ]
    lines += module_hierarchy(model)
    lines += [
        "```",
        "",
        "Module types in use:",
        "",
    ]
    types = sorted({type(m).__name__ for _, m in model.named_modules()})
    lines += [f"- `{t}`" for t in types]
    lines += [
        "",
        "## 3. Operator inventory",
        "",
        f"{len(log.counts)} distinct aten operators, captured with `TorchDispatchMode` on a real",
        "forward pass. Counts are for the sample input above; shapes are one representative call.",
        "",
        "`torch.fx.symbolic_trace` cannot produce this: it raises",
        "`TypeError: torch.finfo() requires a floating point input type` on the Proxy dtype in",
        "`build_extended_attention_mask`, and the MoE expert loop branches on tensor values.",
        "",
    ]
    op_rows = [
        (f"`{name}`", count, f"`{_fmt_shapes(log.shapes[name][0])}`", f"`{_fmt_shape(log.shapes[name][1])}`")
        for name, count in sorted(log.counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    lines += _md_table(["operator", "calls", "input shapes (representative)", "output shape"], op_rows)
    lines += [
        "",
        "### Operators the dense MoE formulation removes",
        "",
        "`aten.nonzero`, `aten.index`, `aten.index_add_` and `aten._local_scalar_dense` come from",
        "the upstream ragged expert loop, which gathers each expert's tokens by value. They are",
        "data-dependent, which is why this model is not `torch.fx`-traceable and why the loop has",
        "no direct device equivalent. `NomicExperts.dense_forward` replaces all of them with two",
        "broadcast-batch matmuls, a multiply and a reduce, and is asserted equal to the loop in",
        "`tests/pcc/test_reference_modules.py`. That is the formulation the TTNN port uses.",
        "",
        "## 4. Per-module tensor shapes",
        "",
        "One row per module invocation, in execution order, for the sample input. Sequence-length",
        f"axes scale with S; here S = {SAMPLE_SEQLEN}. Encoder layers 2 to"
        f" {config.num_hidden_layers - 1} repeat the layer 0 and layer 1 rows verbatim and are",
        "elided.",
        "",
    ]
    lines += _md_table(
        ["module", "type", "input shapes", "output shape"],
        [(f"`{n}`", t, f"`{i}`", f"`{o}`") for n, t, i, o in module_rows],
    )
    lines += [
        "",
        "## 5. Parameters and memory",
        "",
    ]
    lines += _md_table(
        ["group", "parameters", "fp32 (MB)", "bf16 (MB)"],
        [
            (f"`{g}`" if g != "total" else "**total**", f"{n:,}", f"{f32:.1f}", f"{b16:.1f}")
            for g, n, f32, b16 in param_rows
        ],
    )
    lines += [""]
    lines += _para(
        f"Resident weights dominate: {total_params:,} parameters, "
        f"{total_params * BYTES_PER_FP32 / 1e6:.0f} MB at fp32 and "
        f"{total_params * BYTES_PER_BF16 / 1e6:.0f} MB at bf16. Activations are small by "
        f"comparison: one hidden-state tensor at the sample shape is "
        f"{activation_bytes / 1e6:.2f} MB at fp32."
    )
    lines += [""]
    lines += _para(
        "The MoE layers are the exception. The dense all-experts formulation materialises "
        "`(num_experts, tokens, ffn_hidden)` intermediates, which at "
        f"B*S = {SAMPLE_BATCH * SAMPLE_SEQLEN} is "
        f"{_moe_intermediate_mb(config, SAMPLE_BATCH * SAMPLE_SEQLEN):.2f} MB at bf16 and scales "
        f"linearly with token count. At the {MAX_SEQLEN}-token maximum sequence length it is "
        f"{_moe_intermediate_mb(config, MAX_SEQLEN):.0f} MB per MoE layer, which is what the "
        "Phase 1 memory budget has to account for."
    )
    lines += [""]
    return "\n".join(lines)


def _moe_intermediate_mb(config: NomicMoEConfig, tokens: int) -> float:
    """Bytes for one dense all-experts intermediate, the largest transient in the MoE path."""
    return config.num_experts * tokens * config.intermediate_size * BYTES_PER_BF16 / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH, help="markdown output path")
    parser.add_argument("--dot", type=Path, default=None, help="also write the module graph as DOT")
    args = parser.parse_args()

    config = load_vendored_config()
    model = build_synthetic_model(config, seed=0)
    input_ids, attention_mask = random_input_ids(
        SAMPLE_BATCH, SAMPLE_SEQLEN, config, seed=0, pad_lengths=SAMPLE_PAD_LENGTHS
    )

    log = capture_operators(model, input_ids, attention_mask)
    module_rows = capture_module_shapes(model, input_ids, attention_mask)
    param_rows = parameter_table(model)

    args.out.write_text(build_markdown(config, model, log, module_rows, param_rows))
    print(f"wrote {args.out} ({len(log.counts)} operators, {len(module_rows)} module invocations)")

    if args.dot:
        args.dot.write_text(dot_graph(config))
        print(f"wrote {args.dot}; render with: dot -Tpdf {args.dot} -o model_graph.pdf")


if __name__ == "__main__":
    main()
