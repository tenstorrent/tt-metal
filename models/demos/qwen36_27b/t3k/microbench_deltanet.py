# SPDX-License-Identifier: Apache-2.0
"""
Device-bound microbench: isolate where a DeltaNet decode layer's time goes.
Times each component (input projections, the fused DeltaNet recurrence kernel,
out_proj, and an MLP for reference) by enqueuing N iterations and synchronizing
once — amortizing host dispatch to expose device-bound per-call time.

  python3 microbench_deltanet.py --iters 200
"""
import argparse, time
import torch
import ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
from models.demos.qwen36_27b.tt.load_weights import create_dummy_state_dict
from models.demos.qwen36_27b.tt.deltanet import TtGatedDeltaNet, TtDeltaNetState
from models.demos.qwen36_27b.tt.mlp import TtMLP


def bench(name, fn, dev, iters, warmup=10):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {name:28s} {ms:7.3f} ms/call", flush=True)
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    cfg = Qwen36ModelConfig()
    H = cfg.hidden_size
    sd = create_dummy_state_dict(cfg, num_layers=1)  # layer 0 = linear_attention
    dev = ttnn.open_device(device_id=0)
    try:
        dn = TtGatedDeltaNet(dev, sd, 0, cfg)
        mlp = TtMLP(dev, sd, 0, cfg)
        state = TtDeltaNetState(1, cfg.layer_types[:1], dev, cfg)

        x = ttnn.from_torch(torch.randn(1, 1, 1, H) * 0.1, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev)

        print(f"[config] hidden={H} conv_dim(qkv)={dn.key_dim*2+dn.value_dim} "
              f"z={dn.value_dim} v_heads={dn.num_v_heads} k_heads={dn.num_k_heads}", flush=True)
        print(f"[iters={args.iters}] device-bound per-call times:", flush=True)

        # input projections
        bench("in_proj_qkv (5120->%d)" % (dn.key_dim*2+dn.value_dim),
              lambda: ttnn.linear(x, dn.in_proj_qkv_w), dev, args.iters)
        bench("in_proj_z   (5120->%d)" % dn.value_dim,
              lambda: ttnn.linear(x, dn.in_proj_z_w), dev, args.iters)

        # the fused DeltaNet recurrence kernel ALONE
        qkv = ttnn.linear(x, dn.in_proj_qkv_w)
        z = ttnn.linear(x, dn.in_proj_z_w)
        b_proj = ttnn.linear(x, dn.in_proj_b_w)
        a_proj = ttnn.linear(x, dn.in_proj_a_w)
        conv_state = state.get_conv_state(0)
        rec_state = state.get_recurrent_state(0)
        op = ttnn.experimental.deltanet_decode_full

        def run_kernel():
            op(qkv, z, b_proj, a_proj, conv_state, rec_state,
               dn.conv1d_weight_tt, dn.A_log_bf16, dn.dt_bias_bf16, dn.norm_weight,
               num_heads=dn.num_v_heads, num_k_heads=dn.num_k_heads,
               k_head_dim=dn.head_k_dim, v_head_dim=dn.head_v_dim,
               conv_dim=dn.key_dim*2+dn.value_dim, conv_kernel_size=dn.conv_kernel_size,
               head_expand_ratio=dn.head_expand_ratio)
        bench("deltanet_decode_full KERNEL", run_kernel, dev, args.iters)

        # whole DeltaNet token-mixer (projections + kernel + out_proj)
        bench("FULL deltanet layer mixer", lambda: dn._decode_step_full_fused(x, state), dev, args.iters)

        # MLP for reference (same in every layer)
        bench("MLP (gate/up/silu/mul/down)", lambda: mlp(x), dev, args.iters)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
