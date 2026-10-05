# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Speed-of-light (roofline) decode and prefill for Laguna on a QuietBox 2 (4 Blackhole chips).

Applies the formulas of "All About Transformer Inference" (How to Scale Your Model, jax-ml.github.io/scaling-book/
inference/) to Laguna's real tensor shapes (read from the checkpoint's safetensors headers) and the number formats the
TT implementation stores them in (doc/datatype_sweep/selected_precision_config.json):

  decode step   = B * KV_bytes / W + max(2 * B * P_active / C, P_bytes_read / W)     (chapter: "Theoretical Step Time")
  KV bytes      = 2 * bytes_per_value * H * K * L * T                                (chapter: KV cache size)
  prefill       = max(FLOPs / C, weight bytes / W), FLOPs = 2 * P_active * T + attention FLOPs
  TP comms      = 4 * B * D / (3 * W_ici) per layer                                  (chapter: 1D model sharding)

W = DRAM bandwidth, C = peak FLOP/s, B = batch (sequences decoding together), T = tokens, H = head dim, K = KV heads,
L = layers, D = hidden size. Laguna-specific inputs: only 12 of 48 layers keep every past token (the other 36 keep a
512-token sliding window, a fixed cost per sequence), and an MoE layer reads only the experts its tokens picked (10 of
256 per token), so P_bytes_read grows with B. Run:  python models/demos/laguna/demo/roofline.py [--model ...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]

# ---- QuietBox 2 hardware (per Blackhole chip; 4 chips) ------------------------------------------------------------
CHIPS = 4
# DRAM: tt-smi reports dram_speed 16G (16 Gbit/s per pin); 256-bit GDDR6 bus (8 channels in
# tt_metal/soc_descriptors/blackhole_140_arch.yaml) -> 16e9 * 256 / 8 = 512 GB/s, Tenstorrent's published P150 figure.
DRAM_BYTES_PER_S = 16e9 * 256 / 8
# Compute: tech_reports/GEMM_FLOPS/GEMM_FLOPS.md - one matrix engine does 8x16 x 16x16 per cycle = 2*8*16*16 = 4096
# FLOPs/cycle at LoFi (Laguna's matmul fidelity), 1.35 GHz, 13x10 = 130 Tensix cores available for compute.
FLOPS_PER_S = 130 * 4096 * 1.35e9
# Bytes per stored value. Block-float formats store 16 values with one shared 8-bit exponent.
BYTES = {"bf16": 2.0, "bfp8": (16 * 8 + 8) / 8 / 16, "bfp4": (16 * 4 + 8) / 8 / 16}


def checkpoint_dir(model: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, local_files_only=True, allow_patterns=["*.json", "*.safetensors"]))


def tensor_shapes(snapshot: Path) -> dict[str, list[int]]:
    """Shapes of every tensor, read from the safetensors headers only (no weights are loaded)."""
    shapes = {}
    for path in sorted(snapshot.glob("*.safetensors")):
        with open(path, "rb") as handle:
            header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
        shapes.update({name: meta["shape"] for name, meta in header.items() if name != "__metadata__"})
    return shapes


def precision_policy() -> dict[str, str]:
    path = MODEL_DIR / "doc" / "datatype_sweep" / "selected_precision_config.json"
    return json.loads(path.read_text())["policy"]


def model_numbers(snapshot: Path) -> dict:
    config = json.loads((snapshot / "config.json").read_text())
    shapes = tensor_shapes(snapshot)
    policy = precision_policy()
    numel = lambda name: math.prod(shapes[name])  # noqa: E731
    groups = {"experts": 0, "attention": 0, "shared": 0, "dense": 0, "router": 0, "lm_head": 0}
    for name in shapes:
        if ".mlp.experts." in name:
            groups["experts"] += numel(name)
        elif ".self_attn." in name and name.endswith("proj.weight"):
            groups["attention"] += numel(name)
        elif ".mlp.shared_expert." in name:
            groups["shared"] += numel(name)
        elif name.endswith(".mlp.gate.weight"):
            groups["router"] += numel(name)
        elif ".mlp." in name:  # dense feed-forward (layer 0)
            groups["dense"] += numel(name)
        elif name == "lm_head.weight":
            groups["lm_head"] += numel(name)
    dtype = {
        "experts": policy["moe_ff13"],
        "attention": policy["attn_qkv"],
        "shared": policy["shared_ff13"],
        "dense": policy["dense_ff13"],
        "router": policy["router"],
        "lm_head": policy["lm_head"],
    }
    experts, top_k = int(config["num_experts"]), int(config["num_experts_per_tok"])
    layers = int(config["num_hidden_layers"])
    head_dim, kv_heads = int(config["head_dim"]), int(config["num_key_value_heads"])
    types = config["layer_types"]
    heads = [shapes[f"model.layers.{i}.self_attn.q_proj.weight"][0] // head_dim for i in range(layers)]
    full = [i for i in range(layers) if types[i] == "full_attention"]
    sliding = [i for i in range(layers) if types[i] != "full_attention"]
    kv_bytes = BYTES[policy["kv_cache"]]
    other = ("attention", "shared", "dense", "router", "lm_head")
    return {
        "config": config,
        "groups": groups,
        "dtype": dtype,
        "experts": experts,
        "top_k": top_k,
        "moe_layers": sum(1 for name in shapes if name.endswith(".mlp.gate.weight")),
        "expert_bytes_total": groups["experts"] * BYTES[dtype["experts"]],
        "other_bytes": sum(groups[g] * BYTES[dtype[g]] for g in other),
        "active_params": groups["experts"] * top_k / experts + sum(groups[g] for g in other),
        "full_layers": full,
        "sliding_layers": sliding,
        "heads": heads,
        "head_dim": head_dim,
        "window": int(config["sliding_window"]),
        # chapter: KV cache size = 2 * bytes per float * H * K * L * T
        "kv_per_token_full": 2 * kv_bytes * head_dim * kv_heads * len(full),
        "kv_sliding_per_token": 2 * kv_bytes * head_dim * kv_heads * len(sliding),  # x min(context, window)
        "hidden": int(config["hidden_size"]),
    }


def expert_bytes_read(m: dict, tokens: int) -> float:
    """Expert weights one forward pass reads when ``tokens`` tokens each pick top_k of the experts (uniform routing):
    each expert is used with probability 1 - (1 - k/E)^tokens."""
    used_fraction = 1.0 - (1.0 - m["top_k"] / m["experts"]) ** tokens
    return m["expert_bytes_total"] * used_fraction


def decode_step(m: dict, batch: int, context: int) -> dict:
    """Chapter: step = B * KV / W + max(2 * B * P_active / C, P_bytes / W)."""
    W, C = DRAM_BYTES_PER_S * CHIPS, FLOPS_PER_S * CHIPS
    kv = batch * (m["kv_per_token_full"] * context + m["kv_sliding_per_token"] * min(context, m["window"]))
    weights = m["other_bytes"] + expert_bytes_read(m, batch)
    compute_s = 2 * batch * m["active_params"] / C
    memory_s = weights / W
    step = kv / W + max(compute_s, memory_s)
    return {"step_ms": step * 1e3, "kv_gb": kv / 1e9, "weights_gb": weights / 1e9, "compute_ms": compute_s * 1e3,
            "bound": "compute" if compute_s > memory_s else "memory"}  # fmt: skip


def prefill(m: dict, tokens: int) -> dict:
    """max(FLOPs / C, weight bytes / W); attention FLOPs = 4 * heads * head_dim per (query, key) pair per layer
    (scores q.k and the value mix, each a multiply and an add per element); a causal full-attention query sees on average
    half the prompt, a sliding-window query at most ``window`` tokens."""
    W, C = DRAM_BYTES_PER_S * CHIPS, FLOPS_PER_S * CHIPS
    linear = 2 * m["active_params"] * tokens
    pairs_full = tokens * (tokens + 1) / 2
    pairs_sliding = sum(min(t + 1, m["window"]) for t in range(tokens)) if tokens <= 4 * m["window"] else (
        m["window"] * (m["window"] + 1) / 2 + (tokens - m["window"]) * m["window"])  # fmt: skip
    attention = sum(4 * m["heads"][i] * m["head_dim"] * pairs_full for i in m["full_layers"]) + sum(
        4 * m["heads"][i] * m["head_dim"] * pairs_sliding for i in m["sliding_layers"]
    )
    weights = m["other_bytes"] + expert_bytes_read(m, tokens)
    compute_s = (linear + attention) / C
    memory_s = weights / W
    return {"ttft_s": max(compute_s, memory_s), "compute_s": compute_s, "memory_s": memory_s,
            "attention_share": attention / (linear + attention), "pflop": (linear + attention) / 1e15,
            "bound": "compute" if compute_s > memory_s else "memory"}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.environ.get("HF_MODEL", "poolside/Laguna-S-2.1"))
    parser.add_argument(
        "--link-gbps",
        type=float,
        default=100.0,
        help="chip-to-chip Ethernet bandwidth per direction per link, Gbit/s (default 100: the per-link figure in "
        "tech_reports/EthernetMultichip/BasicEthernetGuide.md, which is for Wormhole; no Blackhole figure is documented)",
    )
    parser.add_argument("--links", type=int, default=2, help="links per chip pair used by the ring (Laguna uses 2)")
    args = parser.parse_args()

    m = model_numbers(checkpoint_dir(args.model))
    W, C = DRAM_BYTES_PER_S * CHIPS, FLOPS_PER_S * CHIPS
    print(f"# Speed of light: {args.model} on a QuietBox 2 ({CHIPS} Blackhole chips)\n")
    print(f"Hardware: DRAM {W / 1e12:.2f} TB/s, compute {C / 1e15:.2f} PFLOP/s (LoFi), {CHIPS} chips\n")
    print("| Weight group | Parameters | Stored as | GB on device | Used per token |")
    print("|---|---:|---|---:|---:|")
    for group, count in m["groups"].items():
        used = count * m["top_k"] / m["experts"] if group == "experts" else count
        print(f"| {group} | {count / 1e9:.3f}B | {m['dtype'][group]} | {count * BYTES[m['dtype'][group]] / 1e9:.2f} | "
              f"{used / 1e9:.3f}B |")  # fmt: skip
    print(f"\nActive parameters per token: {m['active_params'] / 1e9:.2f}B. "
          f"KV cache: {m['kv_per_token_full'] / 1e3:.1f} KB per context token ({len(m['full_layers'])} full-attention "
          f"layers) + {m['kv_sliding_per_token'] / 1e3:.1f} KB per context token up to the {m['window']}-token window "
          f"({len(m['sliding_layers'])} sliding layers; at most {m['kv_sliding_per_token'] * m['window'] / 1e6:.1f} MB per "
          "sequence).\n")  # fmt: skip

    print("## Decode\n\nstep = B x KV / W + max(2 x B x P_active / C, weight bytes read / W)\n")
    print("| Context per sequence | Batch | Weights read | KV read | Step | tok/s per user | tok/s total | Limit |")
    print("|---:|---:|---:|---:|---:|---:|---:|---|")
    for context in (128, 4096, 32768, 131072, 1048576):
        for batch in (1, 8, 32):
            if batch > 1 and context > 131072:
                continue
            d = decode_step(m, batch, context)
            print(f"| {context:,} | {batch} | {d['weights_gb']:.1f} GB | {d['kv_gb']:.2f} GB | {d['step_ms']:.2f} ms | "
                  f"{1e3 / d['step_ms']:.0f} | {batch * 1e3 / d['step_ms']:.0f} | {d['bound']} |")  # fmt: skip
    print(f"\nCritical batch (chapter: B_crit = beta x alpha): alpha = C / W = {C / W:.0f} FLOPs per byte. An expert stored "
          f"as {m['dtype']['experts']} ({BYTES[m['dtype']['experts']]} B/weight) becomes compute-bound at "
          f"{C / W * BYTES[m['dtype']['experts']] / 2:.0f} tokens; with top-{m['top_k']} of {m['experts']} that needs "
          f"about {C / W * BYTES[m['dtype']['experts']] / 2 * m['experts'] / m['top_k']:,.0f} sequences per step.\n")  # fmt: skip

    print("## Prefill (time to first token, batch 1)\n\nTTFT >= max((2 x P_active x T + attention FLOPs) / C, "
          "weight bytes read / W)\n")  # fmt: skip
    print("| Prompt tokens | Weights read time | FLOPs | Attention share | Compute time | TTFT floor | Limit |")
    print("|---:|---:|---:|---:|---:|---:|---|")
    for tokens in (128, 1024, 8192, 32768, 131072, 1048576):
        p = prefill(m, tokens)
        print(f"| {tokens:,} | {p['memory_s'] * 1e3:.1f} ms | {p['pflop']:.3f} PFLOP | {p['attention_share'] * 100:.0f}% | "
              f"{p['compute_s'] * 1e3:,.1f} ms | {p['ttft_s'] * 1e3:,.1f} ms | {p['bound']} |")  # fmt: skip

    link = args.link_gbps * 1e9 / 8 * args.links
    per_layer = 4 * 1 * m["hidden"] * BYTES["bf16"] / (3 * link)
    layers = len(m["heads"])
    print(f"\n## Chip-to-chip communication (chapter, 1D model sharding)\n\nper layer = 4 x B x D x bytes / (3 x W_ici) = "
          f"{per_layer * 1e6:.2f} us at B=1, D={m['hidden']}, bf16, W_ici = {args.links} x {args.link_gbps:.0f} Gbit/s; "
          f"x {layers} layers = {per_layer * layers * 1e3:.3f} ms per decode token. This term counts bandwidth only; "
          "at batch 1 each transfer is a few KB, so per-transfer latency (not in the chapter) dominates.\n")  # fmt: skip

    print("## Beyond the chapter\n")
    E, per_chip = m["experts"], m["experts"] // CHIPS
    # Each chip holds per_chip experts; the chip holding the most of a token's top-k experts sets the pace. Expected
    # busiest share under uniform routing, estimated by sampling.
    rng = random.Random(0)
    samples = 200000
    busiest = 0
    for _ in range(samples):
        picked = rng.sample(range(E), m["top_k"])  # one token's experts
        busiest += max(sum(1 for e in picked if e // per_chip == chip) for chip in range(CHIPS))
    busiest /= samples
    W1 = DRAM_BYTES_PER_S
    expert_one = m["expert_bytes_total"] / E  # one expert's weights summed over every MoE layer
    chip_bytes = m["other_bytes"] / CHIPS + busiest * expert_one
    print(f"- Expert placement: each chip holds {per_chip} of the {E} experts; the busiest chip holds {busiest:.2f} of a "
          f"token's {m['top_k']} on average and sets the pace: {chip_bytes / 1e9:.2f} GB / {W1 / 1e9:.0f} GB/s = "
          f"{chip_bytes / W1 * 1e3:.2f} ms -> {W1 / chip_bytes:.0f} tok/s at short context (batch 1).")  # fmt: skip
    print("- Per-transfer latency: tech_reports/EthernetMultichip/BasicEthernetGuide.md measures about 1 us per ring "
          "hop for a 1 KB packet on Wormhole; Blackhole on the QuietBox 2 is not documented there. Not included above.")


if __name__ == "__main__":
    main()
