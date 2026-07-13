# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Localize the serving-specific decode collapse: paged-KV decode vs internal-cache decode, LONG.

Serving pure decode collapses to multilingual garbage (~pos 104) while the demo internal-cache decode
stays coherent — isolated (by ruling out fused + traced) to the PAGED KV path. This runs BOTH decode
paths teacher-forced with the SAME tokens over many steps (full model) and reports per-step logits PCC,
so the step/position where paged diverges from the internal oracle pinpoints the paged bug
(block-boundary at 64/128, page_table indexing, or paged SDPA-decode).

Run: MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B pytest .../test_paged_vs_internal.py -s
"""
import os
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@parametrize_mesh_tp()
def test_paged_vs_internal(mesh_device):
    from loguru import logger

    os.environ.setdefault("HF_MODEL", model_path())
    mesh_device.enable_program_cache()
    nl = int(os.environ["N_LAYERS"]) if os.environ.get("N_LAYERS") else None
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=512, n_layers=nl)
    vocab = model.args.vocab_size
    T = int(os.environ.get("T", "64"))
    N_DEC = int(os.environ.get("N_DEC", "80"))
    torch.manual_seed(0)
    prompt = torch.randint(0, vocab, (T,)).tolist()
    comp0 = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    # ---- oracle: internal concat KV (demo path), greedy, record per-step logits + fed tokens ----
    model.reset_tp()
    ref_logits = [model.prefill_tp(torch.tensor([prompt], dtype=torch.long), valid_len=T).reshape(-1)[:vocab]]
    ref = [int(torch.argmax(ref_logits[0]))]
    pos = T
    for _ in range(N_DEC):
        lg = model.decode_tp(ref[-1], pos).reshape(-1)[:vocab]
        ref_logits.append(lg)
        ref.append(int(torch.argmax(lg)))
        pos += 1

    # ---- paged KV (serving path), teacher-forced with the SAME tokens ----
    block_size = int(os.environ.get("BLOCK_SIZE", "64"))
    num_blocks = 512 // block_size + 8
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    kv_shape = (num_blocks, model.args.n_local_kv_heads, block_size, model.args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    c_dev = model.prefill_paged(torch.tensor([prompt], dtype=torch.long), page_table, valid_len=T)
    c_logits = [ttnn.to_torch(c_dev, mesh_composer=comp0).reshape(-1, vocab)[0].float()]
    pos = T
    for i in range(N_DEC):
        dev = model.prepare_inputs_decode(
            torch.tensor([[ref[i]]], dtype=torch.int32), torch.tensor([pos], dtype=torch.int32), page_table
        )
        out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        c_logits.append(model.process_output_decode(out, 1).reshape(-1)[:vocab].float())
        pos += 1

    logger.info(f"PVI n_layers={len(model.layers)} T={T} N_DEC={N_DEC} block_size={block_size}")
    for i, (r, c) in enumerate(zip(ref_logits, c_logits)):
        p = _pcc(r, c)
        agree = int(torch.argmax(r) == torch.argmax(c))
        tag = "prefill" if i == 0 else f"decode pos={T + i - 1}"
        if p < 0.99 or i < 3 or i % 8 == 0 or not agree:
            logger.info(f"PVI step{i} {tag}: PCC={p:.5f} argmax_agree={agree}")
    ps = [_pcc(r, c) for r, c in zip(ref_logits, c_logits)]
    logger.info(f"PVI_SUMMARY min_pcc={min(ps):.5f} at_step={ps.index(min(ps))} "
                f"first_below_0.99={next((i for i,p in enumerate(ps) if p<0.99), -1)} last_pcc={ps[-1]:.5f}")
