# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Iterative bring-up driver (not a pytest): prefill + decode PCC vs HF on
synthetic weights for one layer kind. Usage:

    python .../dev_bringup.py [dense|moe] [seq_len] [n_decode]
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3].parent))

import ttnn  # noqa: E402
from models.autoports.zai_org_glm_4_7_flash.tests import utils  # noqa: E402
from models.autoports.zai_org_glm_4_7_flash.tt.functional_decoder import (  # noqa: E402
    FunctionalDecoder,
    PagedCacheConfig,
)


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "moe"
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    n_decode = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    layer_idx = utils.LAYER_KINDS[kind]

    import os

    cfg = utils.hf_config()
    if os.environ.get("GLM47_REAL_WEIGHTS"):
        sd = utils.load_real_layer_state_dict(cfg, layer_idx)
        print("using REAL weights")
    else:
        sd = utils.synth_layer_state_dict(cfg, layer_idx)
    x = utils.synth_activations(cfg, layer_idx, S + n_decode, seed=7)

    ref = utils.hf_forward(cfg, utils.build_hf_layer(cfg, layer_idx, sd), x)
    print(f"HF ref done: {ref.shape}")

    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
    try:
        max_context = 4096
        paged = PagedCacheConfig.for_context(max_context, 1)
        expert_dtype = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}[
            os.environ.get("GLM47_EXPERT_DTYPE", "bf8")
        ]
        dec = FunctionalDecoder.from_state_dict(
            sd,
            hf_config=cfg,
            layer_idx=layer_idx,
            mesh_device=device,
            max_batch_size=1,
            max_context=max_context,
            paged_config=paged,
            prefill_chunk_size=1024,
            expert_dtype=expert_dtype,
        )
        kv_cache = dec.allocate_kv_cache()
        pt_torch = utils.make_page_table(1, paged.max_num_blocks, seed=3)
        page_table = ttnn.from_torch(pt_torch, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

        # ---- prefill ----
        x_tt = ttnn.from_torch(x[:, :S].unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        out = dec.prefill_forward(x_tt, kv_cache=kv_cache, page_table=page_table, user_id=0, seq_len=S)
        out_torch = ttnn.to_torch(out).float()[0, 0]
        p = utils.pcc(ref[0, :S], out_torch)
        print(f"PREFILL kind={kind} S={S} PCC={p:.6f}")

        # cache check
        cache_torch = ttnn.to_torch(kv_cache).float()
        got = utils.gather_user_cache(cache_torch, pt_torch, 0, S, paged.block_size)
        want = utils.torch_latent_cache_reference(cfg, sd, x[0, :S])
        print(f"CACHE PCC={utils.pcc(want, got):.6f}")
        print(
            f"CACHE nope PCC={utils.pcc(want[:, :512], got[:, :512]):.6f} rope PCC={utils.pcc(want[:, 512:], got[:, 512:]):.6f}"
        )
        for nm, sl in (("nope", slice(0, 512)), ("rope", slice(512, 576))):
            print(
                f"  {nm}: want std={want[:, sl].std():.5f} mean={want[:, sl].mean():.5f} | got std={got[:, sl].std():.5f} mean={got[:, sl].mean():.5f}"
            )

        # ---- decode ----
        for i in range(n_decode):
            pos = S + i
            x_step = x[:, pos : pos + 1].unsqueeze(0).permute(0, 2, 1, 3)  # [1,1,B=1,H]
            x_tt_d = ttnn.from_torch(x_step, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            cur_pos = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=device)
            rot_idxs = ttnn.from_torch(torch.tensor([[pos]], dtype=torch.uint32), device=device)
            out_d = dec.decode_forward(
                x_tt_d, kv_cache=kv_cache, page_table=page_table, cur_pos_tensor=cur_pos, rot_idxs=rot_idxs
            )
            got_d = ttnn.to_torch(out_d).float()[0, 0, 0]
            pd = utils.pcc(ref[0, pos], got_d)
            print(f"DECODE pos={pos} PCC={pd:.6f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
