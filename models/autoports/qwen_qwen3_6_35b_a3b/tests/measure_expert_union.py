# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Two MoE routing facts the perf and precision analyses depend on. CPU only, no device.

`ttnn.sparse_matmul` works per 32-token group: every expert selected by *any* token in the group
runs for the whole group. That group-union count, not `top_k`, is the real expert-matmul work in
prefill, and it is the number README section 6 limitation 1 divides FLOPs by, so it is measured
here rather than assumed.

Two numbers are printed:

* `uniform` -- the closed form if routing were uniform and independent:
  `E = num_experts * (1 - (1 - top_k / num_experts) ** 32)`. The sanity check.
* `measured` -- HF's own gate on the synthetic weights and synthetic hidden states the perf
  capture uses, `top_k` per token, union per 32-token group, over a 2048-token prefill (the
  measured prefill shape).

The second fact is the bf16 top-k selection agreement (`work_log.md` §5.1): `ttnn.topk` only
accepts bf16, so expert selection runs on bf16 logits and can pick a different 8th expert.

    python models/autoports/qwen_qwen3_6_35b_a3b/tests/measure_expert_union.py
"""

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tt import reference as ref

GROUP = 32  # sparse_matmul tile group == tile height
SEQ = 2048  # the prefill shape the perf capture measures
LAYER = 3  # a full_attention layer; the MoE block is identical in both kinds


def main():
    torch.set_num_threads(2)  # stays polite next to a device run
    cfg = ref.load_hf_text_config()
    n_expert, top_k = cfg.num_experts, cfg.num_experts_per_tok
    uniform = n_expert * (1 - (1 - top_k / n_expert) ** GROUP)

    layer = ref.build_hf_layer(cfg, LAYER, ref.synthetic_layer_state_dict(LAYER))
    x = ref.synthetic_hidden_states(cfg, 1, SEQ, seed=7)
    with torch.no_grad():
        normed = layer.post_attention_layernorm(x)
        # Qwen3_5MoeTopKRouter.forward returns (router_logits, router_scores, router_indices).
        logits, _, indices = layer.mlp.gate(normed.reshape(-1, cfg.hidden_size))
        chosen = indices.reshape(-1, GROUP * top_k)

    unions = torch.tensor([len(torch.unique(row)) for row in chosen], dtype=torch.float64)
    print(f"experts={n_expert} top_k={top_k} group={GROUP} groups={unions.numel()} seq={SEQ}")
    print(f"GROUP_UNION uniform  = {uniform:.1f}")
    print(f"GROUP_UNION measured = mean {unions.mean():.1f}  min {int(unions.min())}  max {int(unions.max())}")
    print(f"GROUP_UNION work multiplier vs ideal gather-by-expert = {unions.mean() / top_k:.1f}x")

    # Second routing fact, same logits: ttnn.topk only accepts bf16, so expert *selection* runs on
    # bf16 logits. How often does that pick a different set than fp32 would? (README section 5.1 of
    # work_log.md quotes this; measured here so it is reproducible rather than asserted.)
    with torch.no_grad():
        fp32_sel = logits.topk(top_k, dim=-1).indices.sort(dim=-1).values
        bf16_sel = logits.to(torch.bfloat16).float().topk(top_k, dim=-1).indices.sort(dim=-1).values
    same_set = (fp32_sel == bf16_sel).all(dim=-1)
    print(
        f"BF16_TOPK token expert-set agreement = {100.0 * same_set.double().mean():.1f}%  "
        f"({int(same_set.numel() - same_set.sum())} of {same_set.numel()} tokens differ)"
    )


if __name__ == "__main__":
    main()
