"""TTMoEGate.forward for the three gate op paths: the single block generalized op, the two block combine
(512 experts) and the deepseek grouped op. Run under tracy.
Env: GW_B tokens (32), GW_N experts (128), GW_K (8), GW_H hidden (2048), GW_NGROUP (1), GW_FUNC (softmax),
GW_BIAS (0, score correction bias), GW_ITERS (4)."""
import os

import torch


def test_gate_paths_probe(device):
    import ttnn
    from models.common.modules.moe.tt_moe_gate import TTMoEGate
    from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig

    b = int(os.environ.get("GW_B", "32"))
    n = int(os.environ.get("GW_N", "128"))
    k = int(os.environ.get("GW_K", "8"))
    h = int(os.environ.get("GW_H", "2048"))
    ngroup = int(os.environ.get("GW_NGROUP", "1"))
    func = os.environ.get("GW_FUNC", "softmax")
    bias = os.environ.get("GW_BIAS", "0") == "1"
    iters = int(os.environ.get("GW_ITERS", "4"))
    torch.manual_seed(3)
    cfg = TTMoEGateConfig(num_routed_experts=n, select_experts_k=k, hidden_size=h, batch_per_device=b,
                          n_group=ngroup, score_func=func, score_correction_bias=bias)
    gate_bias = torch.randn(n, dtype=torch.float32) * 0.01 if bias else None
    gate = TTMoEGate(device, cfg, torch.randn(h, n, dtype=torch.bfloat16), torch_gate_bias=gate_bias)
    x = ttnn.from_torch(torch.randn(1, 1, b, h, dtype=torch.bfloat16), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(f"\n[gate_paths_probe] B={b} N={n} K={k} H={h} n_group={ngroup} func={func} bias={bias}", flush=True)
    for _ in range(iters):
        w, idx = gate.forward(x)
        ttnn.synchronize_device(device)
    print(f"[gate_paths_probe] OK weights={tuple(w.shape)} indices={tuple(idx.shape)}", flush=True)
