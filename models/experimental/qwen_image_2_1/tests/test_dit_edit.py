# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DiT with a condition image (editing layout) vs the diffusers goldens, using the golden text embeddings and
golden condition latents (encoder components are tested separately).

    python -m pytest models/experimental/qwen_image_2_1/tests/test_dit_edit.py -v -s
"""
import os
import time

import pytest
import torch

import ttnn
from models.experimental.qwen_image_2_1.common import rope as rope_mod
from models.experimental.qwen_image_2_1.common import schedule
from models.experimental.qwen_image_2_1.common.config import GOLDENS_DIR
from models.experimental.qwen_image_2_1.common.device import close_device, open_device
from models.experimental.qwen_image_2_1.common.weights import transformer_ckpt
from models.experimental.qwen_image_2_1.tt.dit import DeviceCond, DiTPrecision, QwenImageDiT

EDIT = os.path.join(GOLDENS_DIR, "edit")
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(EDIT, "denoise_edit.pt")), reason="edit goldens missing"
)


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def test_edit_prefix_and_step():
    te = torch.load(os.path.join(EDIT, "text_encoder_edit.pt"), weights_only=False)
    dn = torch.load(os.path.join(EDIT, "denoise_edit.pt"), weights_only=False)
    ve = torch.load(os.path.join(EDIT, "vae_encode.pt"), weights_only=False)
    dev = open_device()
    try:
        prec = DiTPrecision()
        prec.mm_fidelity = ttnn.MathFidelity.HiFi2  # accuracy check, not speed
        t0 = time.time()
        m = QwenImageDiT(dev, transformer_ckpt(), prec)
        print(f"\nloaded DiT in {time.time()-t0:.1f}s")
        # layout from the goldens
        slot_mask = te["image_pad_mask"][0].bool()  # [1045] VLM slots (1024 True)
        img_shapes = [tuple(x) for x in dn["steps"][0]["img_shapes"][0]]  # [(1,64,64) cond, (1,64,64) target]
        target = img_shapes[-1]
        full_mask = torch.cat([slot_mask, torch.ones(target[1] * target[2] // 4, dtype=torch.bool)])
        segments = rope_mod.segments_from_image_pad_mask(full_mask, img_shapes)
        prefix_segments = segments[:-1]
        print("segments", segments)
        cos, sin = rope_mod.joint_cos_sin(segments)
        n_target = target[1] * target[2]
        P = cos.shape[0] - n_target
        rt_prefix = m.rope_tables(cos[:P], sin[:P])
        rt_img = m.rope_tables(cos[P:], sin[P:])
        text_rows = te["prompt_embeds"][0][~slot_mask]  # [21, 4096]
        cond_lat = (
            ve["latent_normalized"][0, :, 0].reshape(64, -1).t().contiguous()
        )  # [4096, 64] packed like _pack_latents
        # the golden step-0 input is [cond latents ; noise]; check our packing matches
        assert torch.allclose(cond_lat.float(), dn["steps"][0]["hidden_states"][0, :n_target].float(), atol=1e-2)
        cond0 = DeviceCond.from_host(dev, schedule.StepConditioning.make(m.time_cond, 0.0))
        t0 = time.time()
        kv, P2 = m.prefix_kv_segments(prefix_segments, text_rows, cond_lat, rt_prefix, cond0)
        ttnn.synchronize_device(dev)
        print(f"prefix over {P2} tokens: {time.time()-t0:.2f}s")
        assert P2 == P == dn["kv_cache_step0"][0][0].shape[1], (P2, P, dn["kv_cache_step0"][0][0].shape)
        worst = 1.0
        for li in range(0, 32):
            kr, vr = dn["kv_cache_step0"][li]
            kt = ttnn.to_torch(kv[li][0])[0, :, :P].permute(1, 0, 2)
            vt = ttnn.to_torch(kv[li][1])[0, :, :P].permute(1, 0, 2)
            pk, pv = pcc(kt, kr[0]), pcc(vt, vr[0])
            worst = min(worst, pk, pv)
            if li in (0, 1, 15, 31):
                print(
                    f"layer {li:2d}: kv pcc k={pk:.5f} v={pv:.5f}  (text rows k={pcc(kt[:8], kr[0][:8]):.5f}, image rows k={pcc(kt[8:8+4096], kr[0][8:8+4096]):.5f}, tail k={pcc(kt[-13:], kr[0][-13:]):.5f})"
                )
        print(f"worst prefix kv pcc: {worst:.5f}")
        # cached step 1 on the target tokens
        st = dn["steps"][1]
        lat = st["hidden_states"][0, -n_target:]  # target latents after step 0
        t01 = float(st["timestep"][0])
        cond = DeviceCond.from_host(dev, schedule.modulation_rows_bf16_like_reference(m.time_cond, t01))
        x = ttnn.from_torch(
            lat.reshape(1, 1, n_target, 64),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        P_pad = kv[0][0].shape[-2]
        key_mask = ttnn.from_torch(
            m.step_key_mask(n_target, P, P_pad),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        t0 = time.time()
        out = m.step(x, cond, kv, rt_img, P, key_mask=key_mask)
        ttnn.synchronize_device(dev)
        print(f"step with {P}-token prefix (eager, incl. compile): {time.time()-t0:.2f}s")
        t0 = time.time()
        out = m.step(x, cond, kv, rt_img, P, key_mask=key_mask)
        ttnn.synchronize_device(dev)
        print(f"step (eager, warm): {time.time()-t0:.2f}s")
        o = ttnn.to_torch(out).reshape(-1, 64)
        p = pcc(o, st["noise_pred"][0])
        print(f"step 1 velocity pcc vs golden = {p:.5f}")
        assert worst > 0.98 and p > 0.98
    finally:
        close_device(dev)
