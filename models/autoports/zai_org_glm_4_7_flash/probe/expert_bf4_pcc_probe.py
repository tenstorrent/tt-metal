# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-layer 4-bit expert PCC probe for zai-org/GLM-4.7-Flash.

Question this answers: can the routed experts run at bfloat4_b (the only
weight policy under which the 30.6B model fits one 32 GB p150) without
losing the MoE layer's accuracy?

Method
- Real weights: all 64 routed experts + shared expert + gate of layer 1
  (the first MoE layer), read from the bf16 checkpoint shard.
- Inputs: 512 random-normal tokens at hidden=2048, fixed seed. Synthetic,
  so this measures WEIGHT quantization sensitivity, not activation stats.
- Reference: torch fp32 SwiGLU FFN per expert, and the full MoE block
  (sigmoid scores + e_score_correction_bias top-4 selection, norm_topk_prob,
  routed_scaling_factor 1.8, plus shared expert).
- Device arms: every expert dense-evaluated on one Blackhole chip via
  ttnn.linear at weight dtype bfloat8_b and bfloat4_b (activations bf16,
  HiFi2, fp32 dest acc). Routing stays host-fp32 and IDENTICAL across arms,
  so the arms differ only in expert weight dtype.
- Shared expert always runs at bfloat8_b: that is the planned policy
  (experts bf4, everything else bf8).

Outputs: per-expert FFN PCC (min/mean), routed-sum PCC, full-MoE PCC per arm,
written to stdout and doc/probe/results.json.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn

SNAPSHOT = Path(
    "/home/stisi/.cache/huggingface/hub/models--zai-org--GLM-4.7-Flash"
    "/snapshots/7dd20894a642a0aa287e9827cb1a1f7f91386b67"
)
SHARD = SNAPSHOT / "model-00002-of-00048.safetensors"
LAYER = 1
HIDDEN = 2048
NUM_EXPERTS = 64
TOP_K = 4
ROUTED_SCALING = 1.8
NUM_TOKENS = 512
SEED = 1234

OUT_DIR = Path(__file__).resolve().parent.parent / "doc" / "probe"


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return float("nan")
    return float((a @ b) / denom)


def ffn_ref(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """SwiGLU in fp32. gate/up/down are torch-layout [out, in]."""
    return (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)) @ down.T


def main() -> None:
    torch.manual_seed(SEED)
    prefix = f"model.layers.{LAYER}.mlp"

    experts = []
    with safe_open(str(SHARD), framework="pt") as f:
        for e in range(NUM_EXPERTS):
            experts.append(
                tuple(
                    f.get_tensor(f"{prefix}.experts.{e}.{name}_proj.weight").float() for name in ("gate", "up", "down")
                )
            )
        gate_w = f.get_tensor(f"{prefix}.gate.weight").float()  # [64, 2048]
        gate_bias = f.get_tensor(f"{prefix}.gate.e_score_correction_bias").float()  # [64]
        shared = tuple(
            f.get_tensor(f"{prefix}.shared_experts.{name}_proj.weight").float() for name in ("gate", "up", "down")
        )
    print(
        f"loaded layer {LAYER}: {NUM_EXPERTS} experts, expert shapes "
        f"{tuple(experts[0][0].shape)}, shared {tuple(shared[0].shape)}"
    )

    x = torch.randn(NUM_TOKENS, HIDDEN)

    # ---- fp32 reference ----
    ref_expert_out = torch.stack([ffn_ref(x, *w) for w in experts])  # [64, N, H]
    scores = torch.sigmoid(x @ gate_w.T)  # [N, 64]
    sel = torch.topk(scores + gate_bias, TOP_K, dim=-1).indices  # [N, 4]
    picked = scores.gather(-1, sel)
    weights = picked / picked.sum(-1, keepdim=True) * ROUTED_SCALING  # [N, 4]

    def combine(expert_out: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(NUM_TOKENS, HIDDEN, dtype=torch.float64)
        for k in range(TOP_K):
            out += weights[:, k, None].double() * expert_out[sel[:, k], torch.arange(NUM_TOKENS)].double()
        return out.float()

    routed_ref = combine(ref_expert_out)
    shared_ref = ffn_ref(x, *shared)
    moe_ref = routed_ref + shared_ref
    n_used = len(sel.unique())
    print(f"routing: {n_used}/{NUM_EXPERTS} experts selected at least once over {NUM_TOKENS} tokens")

    # ---- host-only signal: bfloat4_b weight round-trip PCC ----
    rt = []
    for t in experts[0] + (shared[0],):
        q = ttnn.to_torch(ttnn.from_torch(t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
        rt.append(pcc(t, q))
    print(f"host bfloat4_b weight round-trip PCC (4 sample matrices): " f"min {min(rt):.6f} mean {sum(rt)/len(rt):.6f}")

    # ---- device arms ----
    device = ttnn.open_device(device_id=0)
    results = {"host_weight_roundtrip_bf4_min": min(rt)}
    try:
        ck = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        x_tt = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        def dev_ffn(w3: tuple[torch.Tensor, ...], wdtype) -> torch.Tensor:
            tt_w = [ttnn.from_torch(m.T.contiguous(), dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=device) for m in w3]
            y = ttnn.multiply(
                ttnn.silu(ttnn.linear(x_tt, tt_w[0], compute_kernel_config=ck)),
                ttnn.linear(x_tt, tt_w[1], compute_kernel_config=ck),
            )
            out = ttnn.linear(y, tt_w[2], compute_kernel_config=ck)
            host = ttnn.to_torch(out).float()
            for t in tt_w:
                ttnn.deallocate(t)
            return host

        shared_dev = dev_ffn(shared, ttnn.bfloat8_b)
        results["shared_bf8_pcc"] = pcc(shared_ref, shared_dev)
        print(f"shared expert @ bfloat8_b: PCC {results['shared_bf8_pcc']:.6f}")

        for arm, wdtype in (("bf8", ttnn.bfloat8_b), ("bf4", ttnn.bfloat4_b)):
            per_expert = []
            dev_out = torch.empty(NUM_EXPERTS, NUM_TOKENS, HIDDEN)
            for e in range(NUM_EXPERTS):
                dev_out[e] = dev_ffn(experts[e], wdtype)
                per_expert.append(pcc(ref_expert_out[e], dev_out[e]))
            per_expert_t = torch.tensor(per_expert)
            routed_dev = combine(dev_out)
            moe_dev = routed_dev + shared_dev
            results[arm] = {
                "per_expert_pcc_min": float(per_expert_t.min()),
                "per_expert_pcc_mean": float(per_expert_t.mean()),
                "worst_expert": int(per_expert_t.argmin()),
                "routed_sum_pcc": pcc(routed_ref, routed_dev),
                "full_moe_pcc": pcc(moe_ref, moe_dev),
            }
            r = results[arm]
            print(
                f"experts @ {arm}: per-expert PCC min {r['per_expert_pcc_min']:.6f} "
                f"(expert {r['worst_expert']}) mean {r['per_expert_pcc_mean']:.6f} | "
                f"routed-sum PCC {r['routed_sum_pcc']:.6f} | full-MoE PCC {r['full_moe_pcc']:.6f}"
            )
    finally:
        ttnn.close_device(device)

    results["meta"] = {
        "model": "zai-org/GLM-4.7-Flash",
        "layer": LAYER,
        "tokens": NUM_TOKENS,
        "seed": SEED,
        "inputs": "randn synthetic (weight-quantization probe, not activation-calibrated)",
        "compute": "HiFi2, fp32_dest_acc, packer_l1_acc; activations bfloat16",
        "routing": "host fp32 sigmoid+bias top-4, norm, scale 1.8, identical across arms",
        "shared_expert": "bfloat8_b in all arms (planned policy)",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
