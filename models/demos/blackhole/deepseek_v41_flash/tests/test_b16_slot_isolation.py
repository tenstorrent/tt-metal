# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Slot isolation probe (no scheduler, no vLLM): the SAME token ids (a 4,934-token tool-calling prompt whose correct greedy answer is a get_user_details call with
user_id omar_rossi_1241) are prefilled into every user slot / alone in one slot, then decoded greedily; the per-slot outputs and first-token logits are compared.

Env: ISO_IDS (json {"ids": [...]}), ISO_B (batch, default 16), ISO_CTX (model context, default 262144), ISO_CHUNK (prefill chunk, default auto), ISO_PHASES
(comma list: all, s<k> = ids only in slot k with a short filler prompt in the others, e.g. all,s0,s5,s15), ISO_NTOK (decoded tokens, default 60), ISO_LOGITS=1
(also compare first-token logits across slots in phase 'all').
"""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingParams
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.demos.blackhole.deepseek_v41_flash.tt.generator import Generator
from models.tt_transformers.tt.common import PagedAttentionConfig

IDS = os.environ.get("ISO_IDS", "/mnt/tt-data/ssinghal/dsv4-logs/evals_gpqa_tau3/probe_t22_k3_ids.json")
GOOD = "omar_rossi_1241"


@pytest.mark.timeout(14400)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_TRACE_REGION", "800000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_b16_slot_isolation(mesh_device, device_params):
    B = int(os.environ.get("ISO_B", "16"))
    ctx = int(os.environ.get("ISO_CTX", "262144"))
    chunk = int(os.environ["ISO_CHUNK"]) if os.environ.get("ISO_CHUNK") else None
    ntok = int(os.environ.get("ISO_NTOK", "60"))
    phases = os.environ.get("ISO_PHASES", "all").split(",")
    ids = json.load(open(IDS))["ids"]
    L = len(ids)
    U = B // mesh_device.shape[0]
    a, _, b = os.environ.get("DSV41_LAYERS", "0-39").partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    pp = default_page_params(ctx, U)
    paged = PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"])
    args, model, _, _ = create_tt_model(mesh_device, B, ctx, paged, layer_ids=layer_ids, log=lambda m: logger.info(m))
    gen = Generator([model], [args], mesh_device, tokenizer=args.tokenizer)
    tok = args.tokenizer
    filler = args.encode_prompt("What is the capital of France? Answer in one word.", instruct=True)
    sp = SamplingParams(temperature=0, top_k=1, top_p=1.0)
    results = {}

    def run(name, rows, pad=0):
        """rows: list of B id lists. -> per-slot decoded text, first-token logits [B, V] (optional)"""
        lens = [len(r) for r in rows]
        Lm = max(max(lens), pad)
        toks = torch.zeros(B, Lm, dtype=torch.long)
        for i, r in enumerate(rows):
            toks[i, : len(r)] = torch.tensor(r)
        kw = dict(prompt_lens=lens, sampling_params=sp, enable_trace=True, chunk=chunk)
        gen.prefill_forward_text(toks, **kw)  # compile / trace capture (idempotent)
        first, _ = gen.prefill_forward_text(toks, **kw)
        first = first.view(-1)
        lg = None
        if os.environ.get("ISO_LOGITS") == "1":
            _, lg = gen.prefill_forward_text(toks, return_logits=True, **kw)
            lg = lg.float()
            first = lg.argmax(-1)
        outs = [[int(first[u])] for u in range(B)]
        cur = torch.tensor(lens)
        out_tok = first
        for it in range(ntok):
            out_tok, _ = gen.decode_forward(
                out_tok,
                cur,
                enable_trace=True,
                sampling_params=sp,
                reload_inputs=it == 0,
                reload_page_table=False,
                reload_sampling_params=False,
                reset_sampling_state=it == 0,
            )
            cur = cur + 1
            for u in range(B):
                outs[u].append(int(out_tok[u]))
        gen.m.release_trace()
        texts = [tok.decode(o) for o in outs]
        return outs, texts, lg

    for ph in phases:
        pad = 0
        if "@" in ph:  # e.g. all@8192: right-pad the token tensor to 8192 columns (S_pad of the prefill)
            ph, _, pd = ph.partition("@")
            pad = int(pd)
        if ph == "all":
            rows = [ids] * B
            watch = list(range(B))
        elif ph.startswith("s"):
            k = int(ph[1:])
            rows = [ids if u == k else filler for u in range(B)]
            watch = [k]
        else:
            raise ValueError(ph)
        outs, texts, lg = run(ph, rows, pad)
        for u in watch:
            ok = GOOD in texts[u]
            logger.info(
                f"ISO[{ph}@{pad}] slot {u} (row {u // U}, in-row {u % U}) ok={ok} tokens={outs[u][:24]} text={texts[u][:160]!r}"
            )
            results[(f"{ph}@{pad}", u)] = ok
        if ph == "all":
            ref = outs[0]
            logger.info(
                f"ISO[all] slots identical to slot 0 (first {ntok + 1} tokens): {[int(o == ref) for o in outs]}"
            )
            if lg is not None:
                d = [float((lg[u] - lg[0]).abs().max()) for u in range(B)]
                logger.info(f"ISO[all] first-token logits max|slot u - slot 0|: {[round(x, 3) for x in d]}")
                top = [lg[u].topk(3).indices.tolist() for u in range(B)]
                logger.info(f"ISO[all] first-token top3 ids per slot: {top}")
    logger.info("ISO_SUMMARY " + json.dumps({f"{k[0]}/{k[1]}": v for k, v in results.items()}))
