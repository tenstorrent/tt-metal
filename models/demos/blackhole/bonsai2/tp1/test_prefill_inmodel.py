# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""C's SCALE16 prefill debug: the model's OWN ternary objects (layer 0 GDN in/out, layer 0 MLP, layer 3 attn qkv/o)
through prefill_linear / prefill_mlp at S rows, against float64 from the GGUF codes and scales.

The per-layer dump showed GDN layer 0's in-model prefill output at embedding scale under BONSAI_SCALE16=1 (its
projections ~0), while check_linear's freshly built TernaryLinears pass. This checks the objects the model built.
"""
import os

import numpy as np
import pytest
import torch


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "num_command_queues": 2}], indirect=True)
def test_prefill_inmodel(mesh_device):
    import ttnn
    from models.demos.blackhole.bonsai2.native import cpu_ref
    from models.demos.blackhole.bonsai2.native.a3 import linear as al
    from models.demos.blackhole.bonsai2.native.a3 import prefill as pf
    from models.demos.blackhole.bonsai2.native.gguf_ternary import read_ternary
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    s = int(os.environ.get("PF_ROWS", "2048"))
    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=72 * 64, enable_mtp=False)
    l0, l3 = model.layers[0], model.layers[3]
    mlp = l0.feed_forward.native
    sh = mlp.sh
    rng = np.random.default_rng(0)

    def dense(codes, scales):
        return np.asarray(codes, np.float64) * np.repeat(np.asarray(scales, np.float64), 128, axis=1)

    def x_of(k):
        x = torch.from_numpy(rng.standard_normal((s, k)).astype(np.float32)).bfloat16()
        return x, ttnn.from_torch(x.reshape(1, s, k), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)

    cases = [("gdn_in L0", l0.attention._native_in, al.gdn_in_rows(sh, 0)),
             ("gdn_out L0", l0.attention._native_out, None),
             ("attn_qkv L3", l3.attention.weights.fused_qkv, al.attn_qkv_rows(sh, 3)),
             ("attn_o L3", l3.attention.weights.o_proj, None)]
    names = {"gdn_out L0": "blk.0.ssm_out.weight", "attn_o L3": "blk.3.attn_output.weight"}
    rc = 0
    for label, tl, rows in cases:
        if rows is None:
            t = read_ternary(sh.gguf, names[label])
            rows = (t.codes, t.scales)
        w = dense(*rows)
        x, xt = x_of(tl.k)
        y = ttnn.to_torch(pf.prefill_linear(tl, xt)).float().numpy().reshape(s, tl.n)
        ref = sh.had.rotate(x.float().numpy()) @ w.T
        p = cpu_ref.pcc(y, ref)
        print(f"INMODEL {label} S={s}: pcc {p:.6f} |y|max {np.abs(y).max():.3e} |ref|max {np.abs(ref).max():.3e} "
              f"scales fp16={getattr(tl.s, 'fp16', None)}", flush=True)
        rc |= not (p > 0.999)
    t = {n: read_ternary(sh.gguf, f"blk.0.ffn_{n}.weight") for n in ("gate", "up", "down")}
    x, xt = x_of(5120)
    y = ttnn.to_torch(pf.prefill_mlp(mlp, xt)).float().numpy().reshape(s, 5120)
    xb = x.float().numpy()
    g = cpu_ref.project_ternary(t["gate"], sh.had, xb)
    u = cpu_ref.project_ternary(t["up"], sh.had, xb)
    ref = cpu_ref.project_ternary(t["down"], sh.had, g / (1 + np.exp(-g)) * u)
    p = cpu_ref.pcc(y, ref)
    print(f"INMODEL mlp L0 S={s}: pcc {p:.6f} |y|max {np.abs(y).max():.3e} |ref|max {np.abs(ref).max():.3e} "
          f"scales fp16={getattr(mlp.s_gu, 'fp16', None)}", flush=True)
    rc |= not (p > 0.999)
    assert not rc, "an in-model ternary prefill projection is wrong"
