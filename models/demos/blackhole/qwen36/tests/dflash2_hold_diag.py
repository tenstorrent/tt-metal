# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: which stage of a join changes a HELD slot's GDN spec state? Snapshots user 0's whole ring
(all T token slots), its conv window row and its attention K blocks after each stage: (1) B's prefill into
slot 1, (2) B's seed replay (A held), (3) a pure hold step (step(only=[1])). Prints per-slot max |delta|.

Run: MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/dflash2_hold_diag.py -v -s
"""
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_dflash2_serving import _prefill
from models.demos.blackhole.qwen36.tests.test_spec_batched import _batch_prompts, _blocks_per_user
from models.demos.blackhole.qwen36.tests.test_spec_lossless import MAX_NEW, NUM_BLOCKS
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

B = 2
K = 7


@run_for_blackhole()
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_hold_diag(mesh_device):
    if not _MULTI:
        pytest.skip("TP only")
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.dflash2_serving import DFlash2ServingDecoder

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=B, max_seq_len=NUM_BLOCKS * BLOCK_SIZE)
    model.set_gdn_fused_decode(True)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = _batch_prompts(2, tokenizer)
    bpu = _blocks_per_user(max(len(p) for p in prompts), K, MAX_NEW + 8)
    page_tables = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])
    kv_shape = [B * bpu, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=B)
    dec = DFlash2ServingDecoder(model, num_blocks=bpu)
    gdns = dec._gdn
    att0 = next(layer.attention for layer in model.layers if layer.is_full_attention)
    T = K + 1

    def snap():
        out = {}
        for li in (0, len(gdns) // 2, len(gdns) - 1):
            dn = gdns[li]
            Nv = dn.Nv
            ring = ttnn.to_torch(ttnn.get_device_tensors(dn._spec_ring)[0])  # [T*B*Nv, Dk, Dv]
            out[f"ring L{li}"] = torch.stack([ring[(t * B + 0) * Nv : (t * B + 0) * Nv + Nv].clone() for t in range(T)])
            out[f"win L{li}"] = ttnn.to_torch(ttnn.get_device_tensors(dn._verify_win_buf)[0])[0].clone()
        out["K blocks"] = ttnn.to_torch(ttnn.get_device_tensors(att0.paged_k)[0])[page_tables[0].tolist()].clone()
        return out

    def diff(tag, a, b):
        for k in a:
            x, y = a[k].float(), b[k].float()
            if x.dim() == 4 and k.startswith("ring"):
                per = [float((x[t] - y[t]).abs().max()) for t in range(x.shape[0])]
                logger.info(
                    f"[hold-diag] {tag}: {k} per-slot max|delta| {['%.2e' % v for v in per]} (mi[0]={dec.mi[0]})"
                )
            else:
                logger.info(f"[hold-diag] {tag}: {k} max|delta| {float((x - y).abs().max()):.3e}")

    try:
        dec.alloc()
        dec.warm()
        f = _prefill(model, dec, 0, prompts[0], page_tables)
        dec.capture(warm_position=len(prompts[0]) + 1)
        dec.begin(0, f, len(prompts[0]), page_tables[0])
        for _ in range(3):
            dec.step()
        dec.end(0)
        fA = _prefill(model, dec, 0, prompts[0], page_tables)
        dec.begin(0, fA, len(prompts[0]), page_tables[0])
        for _ in range(3):
            dec.step()
        ttnn.synchronize_device(device)
        s0 = snap()
        # a plain re-read (device idle) must be identical
        diff("re-read", s0, snap())
        # (0) a pure hold step with A the ONLY live slot: step(only=[]) -> everything holds
        dec.step(only=[])
        ttnn.synchronize_device(device)
        s0b = snap()
        diff("hold step (all held)", s0, s0b)
        fB = _prefill(model, dec, 1, prompts[1], page_tables)
        ttnn.synchronize_device(device)
        s1 = snap()
        diff("after prefill(B)", s0b, s1)
        dec.begin(1, fB, len(prompts[1]), page_tables[1])
        ttnn.synchronize_device(device)
        s2 = snap()
        diff("after begin(1) [A held]", s1, s2)
        dec.step(only=[1])
        ttnn.synchronize_device(device)
        s3 = snap()
        diff("after step(only=[1]) [A held]", s2, s3)
    finally:
        dec.release()
        model.free_kv_caches()
