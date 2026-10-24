# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""G6 developer harness (report only, no assertion): the full model on NATURAL prose (A Tale of Two Cities, no
repetition) at several prompt lengths against an fp32 HF reference, with the bf16 HF reference measured the same way
beside it. One causal HF forward at the longest length serves every prefix. Per length it prints TT-vs-fp32 and
HF-bf16-vs-fp32: PCC over all rows, PCC per 256-row band (worst band), top-1 / top-5 agreement over the last 64
positions and over all rows, the last-row PCC. Env: G6_NATURAL_LENS (default 512,1024,2048,4096), HF_MODEL,
G6_NATURAL_DUMP_DIR (saves the logit tensors).

    HF_MODEL=google/gemma-4-12B-it pytest models/demos/gemma4/tests/unit/test_natural_gate_g6.py -k 1x4 -s
"""

import bz2
import gc
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn

from ...tests.test_factory import parametrize_mesh_with_fabric

TALE = Path(__file__).resolve().parents[4] / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2"


def _pcc(a, b):
    a = a.flatten().double() - a.flatten().double().mean()
    b = b.flatten().double() - b.flatten().double().mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def _agree(x, ref, k, rows):
    top = x[rows].topk(k, -1).indices
    return float((top == ref[rows].argmax(-1).unsqueeze(-1)).any(-1).double().mean())


def _report(tag, x, ref, L):
    bands = [(s, _pcc(x[s : min(s + 256, L)], ref[s : min(s + 256, L)])) for s in range(0, L, 256)]
    worst = min(bands, key=lambda t: t[1])
    last = slice(max(0, L - 64), L)
    every = slice(0, L)
    return (
        f"{tag}: pcc_all {_pcc(x[:L], ref[:L]):.4f} worst_band [{worst[0]},{min(worst[0] + 256, L)}) {worst[1]:.4f} "
        f"last64 top1 {_agree(x, ref, 1, last):.3f} top5 {_agree(x, ref, 5, last):.3f} "
        f"all top1 {_agree(x, ref, 1, every):.3f} top5 {_agree(x, ref, 5, every):.3f} last_row_pcc {_pcc(x[L - 1], ref[L - 1]):.4f}"
    )


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="developer harness: fp32 HF reference on the host")
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_natural_gate(mesh_device, reset_seeds, request):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from models.demos.gemma4.tt.common import create_tt_model

    model_path = os.environ["HF_MODEL"]
    lengths = sorted(int(v) for v in os.environ.get("G6_NATURAL_LENS", "512,1024,2048,4096").split(","))
    lmax = lengths[-1]
    dump_dir = os.environ.get("G6_NATURAL_DUMP_DIR")
    text = bz2.open(TALE).read().decode("utf-8", "replace")
    text = text[text.find("It was the best of times") :]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ids = tokenizer.encode(text[: lmax * 8], return_tensors="pt")[:, :lmax]
    assert ids.shape[1] == lmax, f"only {ids.shape[1]} tokens of prose for {lmax}"
    logger.info(f"natural prompt: {lmax} tokens, head {tokenizer.decode(ids[0, :12])!r}")

    refs = {}
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        hf = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        hf.eval()
        with torch.no_grad():
            refs[name] = hf(ids).logits.float()[0]
        del hf
        gc.collect()
        logger.info(f"HF {name} logits done: {tuple(refs[name].shape)}")
        if dump_dir:
            torch.save(
                {"hf": refs[name], "input_ids": ids}, f"{dump_dir}/{model_path.split('/')[-1]}_natural_hf_{name}.pt"
            )
    print(
        f"RESULT natural_hf model={model_path.split('/')[-1]} "
        + " | ".join(f"len={L} " + _report("hf16_vs_fp32", refs["bf16"], refs["fp32"], L) for L in lengths)
    )

    model_args, tt_model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device=mesh_device, max_batch_size=1, max_seq_len=lmax, model_path=model_path, create_kv_cache=True
    )
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    embed_w = state_dict.get("model.language_model.embed_tokens.weight", state_dict.get("model.embed_tokens.weight"))
    for L in lengths:
        ids_l = ids[:, :L]
        tokens_tt = ttnn.from_torch(
            ids_l.to(torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        embeds = tt_model.embed_tokens(tokens_tt)
        embeds = ttnn.reshape(embeds, (1, 1, L, model_args.hidden_size))
        embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)
        embeds_torch = (
            (F.embedding(ids_l.long(), embed_w) * tt_model.embed_scale).float() if embed_w is not None else None
        )
        tt_logits = tt_model.ttnn_prefill_forward(
            embeds, page_table=None, kv_cache=tt_kv_cache, input_ids_torch=ids_l, embeds_torch=embeds_torch
        )
        x = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0]).float()
        tt_logits.deallocate(True)
        x = x.reshape(-1, x.shape[-1])[:L]
        if dump_dir:
            torch.save({"tt": x, "input_ids": ids_l}, f"{dump_dir}/{model_path.split('/')[-1]}_natural_tt_len{L}.pt")
        print(
            f"RESULT natural model={model_path.split('/')[-1]} len={L} "
            + _report("tt_vs_fp32", x, refs["fp32"], L)
            + " | "
            + _report("hf16_vs_fp32", refs["bf16"], refs["fp32"], L)
            + " | "
            + _report("tt_vs_hf16", x, refs["bf16"], L)
        )
