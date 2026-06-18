# SPDX-License-Identifier: Apache-2.0
"""Run deltanet_decode_full with fixed seeded inputs; print a stable checksum of
the output + new state, for before/after kernel-edit regression checking."""
import torch
import ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import create_dummy_state_dict
from models.demos.qwen36_27b.tt.deltanet import TtGatedDeltaNet, TtDeltaNetState

torch.manual_seed(1234)
cfg = Qwen36ModelConfig()
sd = create_dummy_state_dict(cfg, num_layers=1)
dev = ttnn.open_device(device_id=0)
try:
    dn = TtGatedDeltaNet(dev, sd, 0, cfg)
    state = TtDeltaNetState(1, cfg.layer_types[:1], dev, cfg)
    # seed the recurrent + conv state with non-zero values
    H, Dk, Dv = dn.num_v_heads, dn.head_k_dim, dn.head_v_dim
    rs = torch.randn(1, H, Dk, Dv) * 0.1
    state.recurrent_states[0] = ttnn.from_torch(rs.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    x = ttnn.from_torch((torch.randn(1, 1, 1, cfg.hidden_size) * 0.1).to(torch.bfloat16),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    out = dn._decode_step_full_fused(x, state)
    o = ttnn.to_torch(out).float().reshape(-1)
    ns = ttnn.to_torch(state.get_recurrent_state(0)).float().reshape(-1)
    print(f"OUT  sum={o.sum().item():.6f}  absmean={o.abs().mean().item():.6f}  [:5]={o[:5].tolist()}", flush=True)
    print(f"STATE sum={ns.sum().item():.6f}  absmean={ns.abs().mean().item():.6f}", flush=True)
finally:
    ttnn.close_device(dev)
