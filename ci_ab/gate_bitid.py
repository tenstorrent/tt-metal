#!/usr/bin/env python3
"""Bit identity of the prefill MoE gate (ttnn.experimental.deepseek_prefill.moe_grouped_topk) across builds: runs DeepSeek V3's
gate (256 experts, 8 groups, 4 kept, 8 activated) at 640 and 4096 tokens on the perf cases' inputs and on a tie-heavy input
(scores and bias on a grid of 8 values, so the stable sort orders equal scores by index), and Kimi K3's (896 experts, one
group, 16 activated, on the perf inputs and on the tie-heavy input), stable sort as the models call it and, for DeepSeek V3 at 640 tokens, the unstable network too, and
writes the sha256 of each output's bytes (weights, indices) and the outputs themselves.
usage: gate_bitid.py <out_dir> [case...]"""
import hashlib, os, sys

import torch
import ttnn

CASES = {
    "dsv3_t640": (640, 256, 8, 4, 8, 2.5, "randn"),
    "dsv3_t4096": (4096, 256, 8, 4, 8, 2.5, "randn"),
    "dsv3_t640_ties": (640, 256, 8, 4, 8, 2.5, "ties"),
    "dsv3_t4096_ties": (4096, 256, 8, 4, 8, 2.5, "ties"),
    "kimik3_t640": (640, 896, 1, 1, 16, 1.0, "randn"),
    "kimik3_t640_ties": (640, 896, 1, 1, 16, 1.0, "ties"),
    "dsv3_t640_unstable": (640, 256, 8, 4, 8, 2.5, "randn", False),
    "dsv3_t640_ties_unstable": (640, 256, 8, 4, 8, 2.5, "ties", False),
    # DeepSeek V4 (one group, top-6, sqrtsoftplus, route scale 1.5): Flash's 256 experts and Pro's 384
    "dsv4f_t640": (640, 256, 1, 1, 6, 1.5, "randn", True, "sqrtsoftplus"),
    "dsv4f_t640_ties": (640, 256, 1, 1, 6, 1.5, "ties", True, "sqrtsoftplus"),
    "dsv4p_t640": (640, 384, 1, 1, 6, 1.5, "randn", True, "sqrtsoftplus"),
    # part 11: DeepSeek V3 with each token's winning groups out of group id order and ties across and inside them
    # (gties), stable and unstable; MiniMax M3 (128 experts, one group, top-4, route scale 2, the op's default unstable
    # sort and TILE weights) on the perf and tie-heavy inputs
    "dsv3_t640_gties": (640, 256, 8, 4, 8, 2.5, "gties", True),
    "dsv3_t640_gties_unstable": (640, 256, 8, 4, 8, 2.5, "gties", False),
    "mm3_t640": (640, 128, 1, 1, 4, 2.0, "randn", False, "sigmoid", "tile"),
    "mm3_t640_ties": (640, 128, 1, 1, 4, 2.0, "ties", False, "sigmoid", "tile"),
    # several token rows a core on the one-group path, so a row starts after another row's top-k
    "kimik3_t4096": (4096, 896, 1, 1, 16, 1.0, "randn"),
    "dsv4f_t4096": (4096, 256, 1, 1, 6, 1.5, "randn", True, "sqrtsoftplus"),
}
# moe_hash_gate (DeepSeek V4's hash-routed layers): tokens, experts, k, route scale, vocab, score_func
HASH_CASES = {
    "hash_v4f_t640": (640, 256, 6, 1.5, 2048, "sqrtsoftplus"),
    "hash_v4p_t640": (640, 384, 6, 2.5, 2048, "sqrtsoftplus"),
}


def inputs(tokens, experts, kind):
    torch.manual_seed(7)
    if kind == "gties":
        # per token: groups ranked by a random permutation through expert 1 (and its tie, expert 2), every group's
        # expert 0 at 0.75, the rest on four values below; the winning groups arrive in rank order, not id order
        groups = experts // 32
        sc = (torch.randint(-4, 0, (1, 1, tokens, experts)).float() / 4).view(1, 1, tokens, groups, 32)
        perm = torch.stack([torch.randperm(groups) for _ in range(tokens)]).float().view(1, 1, tokens, groups)
        sc[..., 0] = 0.75
        sc[..., 1] = perm * 0.0625
        sc[..., 2] = perm * 0.0625
        return sc.reshape(1, 1, tokens, experts).to(torch.bfloat16), torch.zeros((1, 1, tokens, experts)).to(torch.bfloat16)
    if kind == "randn":
        sc = torch.randn((1, 1, tokens, experts), dtype=torch.float32)
        bias = 0.1 * torch.randn((1, 1, tokens, experts), dtype=torch.float32)
    else:
        sc = torch.randint(-4, 4, (1, 1, tokens, experts)).float() / 4
        bias = torch.zeros((1, 1, tokens, experts))
    return sc.to(torch.bfloat16), bias.to(torch.bfloat16)


def record(f, out, name, w, idx):
    for tag, x in (("weights", w), ("indices", idx)):
        a = ttnn.to_torch(x).contiguous()
        h = hashlib.sha256(a.view(torch.uint8).numpy().tobytes() if a.dtype != torch.bfloat16
                           else a.view(torch.int16).numpy().tobytes()).hexdigest()
        torch.save(a, os.path.join(out, f"{name}_{tag}.pt"))
        line = f"{name} {tag} {tuple(a.shape)} {a.dtype} {h}"
        print(line, flush=True)
        f.write(line + "\n")


def hash_inputs(tokens, experts, k, vocab):
    torch.manual_seed(7)
    logits = torch.randn((1, 1, tokens, experts), dtype=torch.float32)
    ids = torch.randint(0, vocab, (tokens,), dtype=torch.int64)
    tid2eid = torch.stack([torch.randperm(experts)[:k] for _ in range(vocab)])
    return logits, ids, tid2eid


def run_hash(dev, name):
    tokens, experts, k, scale, vocab, score_func = HASH_CASES[name]
    logits, ids, tid2eid = hash_inputs(tokens, experts, k, vocab)
    tl = ttnn.from_torch(logits, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    ti = ttnn.from_torch(ids.reshape(tokens // 32, 32).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                         device=dev)
    t2 = torch.zeros(vocab, 16, dtype=torch.int16)
    t2[:, :k] = tid2eid.to(torch.int16)
    tt = ttnn.from_torch(t2, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    return ttnn.experimental.deepseek_prefill.moe_hash_gate(tl, ti, tt, n_activated_experts=k, route_scale=scale,
                                                           epsilon=1e-20, score_func=score_func)


def main():
    out = sys.argv[1]
    names = sys.argv[2:] or list(CASES) + list(HASH_CASES)
    os.makedirs(out, exist_ok=True)
    dev = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        with open(os.path.join(out, "sha256.txt"), "w") as f:
            for name in names:
                if name not in HASH_CASES:
                    tokens, experts, n_groups, topk_groups, n_act, scale, kind = CASES[name][:7]
                    stable = CASES[name][7] if len(CASES[name]) > 7 else True
                if name in HASH_CASES:
                    w, idx = run_hash(dev, name)
                    record(f, out, name, w, idx)
                    continue
                score_func = CASES[name][8] if len(CASES[name]) > 8 else "sigmoid"
                tile_weights = len(CASES[name]) > 9 and CASES[name][9] == "tile"
                sc, bias = inputs(tokens, experts, kind)
                t = lambda a: ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                w, idx = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
                    t(sc), t(bias), n_groups=n_groups,
                    summed_experts_per_group=n_groups // topk_groups if n_groups > 1 else 1,
                    topk_groups=topk_groups, n_activated_experts=n_act, route_scale=scale, stable_sort=stable,
                    epsilon=1e-20, score_func=score_func,
                    weights_layout=ttnn.ROW_MAJOR_LAYOUT if n_act <= 32 and not tile_weights else ttnn.TILE_LAYOUT)
                record(f, out, name, w, idx)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
