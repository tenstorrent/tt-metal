# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""WHOLE-model prefill (40 layers, chunked, paged hand-off) first-token logits with the prefill indexer ON vs OFF (dense attention over every compressed entry) vs the CPU reference
(``reference/ref_prefill_dump.py --head`` dump with final.pt). Env: DSV41_PS_DIR (dump dir with final.pt), DSV41_PS_S, DSV41_PS_C (chunk, default 512), DSV41_LAYERS ("0-39")."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.demos.blackhole.deepseek_v41_flash.tt.generator import Generator
from models.tt_transformers.tt.common import PagedAttentionConfig

S = int(os.environ.get("DSV41_PS_S", "2048"))
C = int(os.environ.get("DSV41_PS_C", "512"))
LAYERS = os.environ.get("DSV41_LAYERS", "0-39")
DIR = os.environ.get("DSV41_PS_DIR", f"/mnt/tt-data/ssinghal/dsv4-prefill-s{S}b1full")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 300_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(10000)
@torch.no_grad()
def test_prefill_sparse_logits(mesh_device):
    os.environ["DSV41_ALLOW_DENSE"] = "1"
    md = mesh_device
    log = lambda m: print(m, flush=True)
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    U = 1
    B = 4 * U
    toks = torch.load(os.path.join(DIR, "tokens.pt"))["prefill_tokens"]
    prompt = toks[:1].expand(B, -1).contiguous()
    assert prompt.shape[1] == S
    fin = torch.load(os.path.join(DIR, "final.pt"))
    ref = fin["prefill_logits"].reshape(-1).float()
    pp = default_page_params(S + 64, U)
    args, model, pool, _ = create_tt_model(
        md,
        B,
        S + 64,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    gen = Generator(model, args, md)
    lens = torch.full((B,), S)
    out = {}
    for tag in ("dense", "sparse", "sparse2"):
        if tag == "sparse":
            model.enable_prefill_sparse(c_max=C)
        t = time.perf_counter()
        lg = gen.prefill_forward_text(prompt, prompt_lens=lens, chunk=C, return_logits=True, enable_trace=False)
        dt = time.perf_counter() - t
        out[tag] = lg[0].reshape(-1).float()
        log(
            f"PSLOGITS {tag}: {dt:.1f} s; PCC vs reference {R.pcc(out[tag], ref):.5f}; argmax {int(out[tag].argmax())} (reference {int(ref.argmax())}); top-5 overlap "
            f"{len(set(out[tag].topk(5).indices.tolist()) & set(ref.topk(5).indices.tolist()))}/5; users equal {bool(all((lg[i].float() - lg[0].float()).abs().max() < 1e-3 for i in range(B)))}"
        )
    log(
        f"PSLOGITS sparse vs dense PCC {R.pcc(out['sparse'], out['dense']):.5f}; dense vs reference {R.pcc(out['dense'], ref):.5f}; sparse vs reference {R.pcc(out['sparse'], ref):.5f}"
    )
