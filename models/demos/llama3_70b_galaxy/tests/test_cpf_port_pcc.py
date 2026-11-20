# Accuracy of the Wormhole prefill levers (gtobar/prefill-levers) on Blackhole, FULL 80-layer model, real text.
# Last-token logits of each variant vs a reference (PCC, top-1, top-5 overlap). One process, one model.
#   cold: one prompt prefilled from position 0; ref = all new levers off.
#     seqsplit  LLAMA_GALAXY_PREFILL_SEQSPLIT (residual add + RMSNorm on seqlen/8 rows)
#     qkvsplit  LLAMA_GALAXY_PREFILL_QKV_SEQSPLIT (QKV heads + RoPE on seqlen/4 rows)
#     pad1024   LLAMA_GALAXY_PREFILL_PAD=1024 (a 2900-token prompt pads to 3072 instead of 4096)
#   warm: P = 2048 cached prefix rows, then the rest of the prompt with start_pos = P; ref = cold prefill of the whole
#   prompt (levers off).
#     chunked   the existing path (chunked SDPA on the owning mesh column)
#     cpf       LLAMA_GALAXY_PREFILL_CPF_RING (ring SDPA over prefix + new rows on all 4 columns)
#     cpf_qkv   + LLAMA_GALAXY_PREFILL_CPF_QKV_SEQSPLIT
#     rep_w / rep_r  + LLAMA_GALAXY_PREFILL_CPF_REPLICA: first call writes the prefix K|V replica, second reads it
#     rep_other  replica valid for another prefix: must write (not read) and still match
# Launch env: source sla_scripts/bh_tp32/env.sh, plus LLAMA_GALAXY_PREFILL_PAD=1024 LLAMA_GALAXY_PREFILL_SEQSPLIT=1
# (both are read when the model is built: CCL buffers for 3072/5120/6144 and the seq-split switch; the test toggles them
# per variant afterwards).
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_cpf_port_pcc.py -s --timeout=0
import glob
import os

import pytest
import torch
from loguru import logger

import ttnn
import models.demos.llama3_70b_galaxy.tt.generator as gen_mod
import models.demos.llama3_70b_galaxy.tt.llama_ccl as ccl_mod
from models.demos.llama3_70b_galaxy.demo.text_demo import create_tt_model
from models.demos.llama3_70b_galaxy.tt.generator import Generator
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM

P = 2048
RUNTIME = [
    "LLAMA_GALAXY_PREFILL_QKV_SEQSPLIT",
    "LLAMA_GALAXY_PREFILL_CPF_RING",
    "LLAMA_GALAXY_PREFILL_CPF_QKV_SEQSPLIT",
    "LLAMA_GALAXY_PREFILL_CPF_REPLICA",
    "LLAMA_BH_PREFILL_NORM_BF8",
    "LLAMA_BH_PREFILL_WO_MMRS",
]
NORM8 = {"LLAMA_BH_PREFILL_NORM_BF8": "1"}
WOMM = {"LLAMA_BH_PREFILL_WO_MMRS": "1"}
PAD_SEQLENS = list(gen_mod.PREFILL_EXTRA_SEQLENS)
SEQSPLIT_AT_LAUNCH = ccl_mod.PREFILL_SEQSPLIT


def _set(env, seqsplit=False, pad=False):
    for k in RUNTIME:
        os.environ.pop(k, None)
    os.environ.update(env)
    ccl_mod.PREFILL_SEQSPLIT = seqsplit
    gen_mod.PREFILL_EXTRA_SEQLENS = PAD_SEQLENS if pad else []


def _cmp(tag, ref, o):
    pcc = torch.corrcoef(torch.stack([ref, o]))[0, 1].item()
    t1 = int(torch.argmax(ref) == torch.argmax(o))
    ov = len(set(torch.topk(ref, 5).indices.tolist()) & set(torch.topk(o, 5).indices.tolist()))
    logger.info(f"CPFPCC {tag:28s} pcc={pcc:.5f} top1_match={t1} top5_overlap={ov}/5")
    return pcc, t1


@pytest.mark.parametrize(
    "device_params",
    [
        {
            TRACE_MODEL_KEY_PARAM: "llama3.3-70b-galaxy",
            "num_command_queues": 1,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "worker_l1_size": 1345000,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_cpf_port_pcc(mesh_device, device_params):
    assert (
        PAD_SEQLENS and SEQSPLIT_AT_LAUNCH
    ), "launch with LLAMA_GALAXY_PREFILL_PAD=1024 LLAMA_GALAXY_PREFILL_SEQSPLIT=1"
    model_args, model, page_table, kv = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=1,
        optimizations=LlamaOptimizations.performance,
        max_seq_len=131072,
        num_layers=int(os.environ.get("CPFPCC_LAYERS", "80")),
        dummy_weights=False,
        page_params={"page_block_size": 64, "page_max_num_blocks": 2048},
        dtype=ttnn.bfloat8_b,
        use_paged_kv_cache=True,
    )
    for name in ("tt_ccl", "tt_ccl_prefill"):
        c = getattr(model, name, None)
        if c is not None:
            fields = ("mode", "use_prefetcher", "is_blackhole", "is_qwen", "use_ring_rs_prefill", "use_ring_ag_prefill")
            logger.info(f"CPFPCC {name}: " + " ".join(f"{f}={getattr(c, f, None)}" for f in fields))
    tok = model_args.create_tokenizer()
    model_args.tokenizer = tok
    gen = Generator(model, model_args, mesh_device, tokenizer=tok)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = "\n\n".join(
        open(f, errors="ignore").read() for f in sorted(glob.glob(os.path.join(root, "**/*.md"), recursive=True))
    )
    text += "\n\n".join(open(f, errors="ignore").read() for f in sorted(glob.glob(os.path.join(root, "tt/*.py"))))
    ids = tok.encode(text) if hasattr(tok, "encode") else tok(text)["input_ids"]
    logger.info(f"CPFPCC corpus tokens {len(ids)}")
    pt = page_table[:1]

    def run(tokens, start=None):
        n = int(tokens.shape[-1])
        out = gen.prefill_forward_text(
            tokens,
            page_table=pt,
            kv_cache=kv,
            prompt_lens=[n],
            enable_trace=False,
            sampling_params=None,
            start_pos=None if start is None else [start],
        )
        out = out[0] if isinstance(out, tuple) else out
        return out.float().reshape(-1)[: model_args.vocab_size]

    bad = []
    only = os.environ.get("CPFPCC_ONLY", "cold,warm").split(",")

    if "cold" in only:
        for n in (2900, 4096, 2048):
            tokens = torch.tensor(ids[:n]).unsqueeze(0)
            _set({})
            ref = run(tokens)
            cold = {
                "ref_again": ({}, False, False),
                "seqsplit": ({}, True, False),
                "qkvsplit": ({"LLAMA_GALAXY_PREFILL_QKV_SEQSPLIT": "1"}, False, False),
                "seq+qkv": ({"LLAMA_GALAXY_PREFILL_QKV_SEQSPLIT": "1"}, True, False),
                "seq+norm8": (NORM8, True, False),
                "seq+womm": (WOMM, True, False),
                "seq+norm8+womm": (NORM8 | WOMM, True, False),
                "norm8": (NORM8, False, False),
            }
            if n == 2900:
                cold["pad1024"] = ({}, False, True)
                cold["pad+seq+qkv"] = ({"LLAMA_GALAXY_PREFILL_QKV_SEQSPLIT": "1"}, True, True)
            for name, (env, ss, pad) in cold.items():
                _set(env, ss, pad)
                pcc, t1 = _cmp(f"cold n={n} {name}", ref, run(tokens))
                if pcc < 0.98 or not t1:
                    bad.append(f"cold n={n} {name} pcc={pcc:.4f} top1={t1}")

    if "warm" in only:
        cpf = {"LLAMA_GALAXY_PREFILL_CPF_RING": "1"}
        cpfq = cpf | {"LLAMA_GALAXY_PREFILL_CPF_QKV_SEQSPLIT": "1", "LLAMA_GALAXY_PREFILL_QKV_SEQSPLIT": "1"}
        rep = cpf | {"LLAMA_GALAXY_PREFILL_CPF_REPLICA": "1"}
        # n = total prompt; new rows n - P pad to 2048 / 3072 (PAD) / 4096
        for n, pad in ((P + 2000, False), (P + 2900, True), (P + 4000, False)):
            tokens = torch.tensor(ids[:n]).unsqueeze(0)
            other = torch.tensor(ids[50000 : 50000 + n]).unsqueeze(0)  # different prefix, same lengths
            _set({}, False, pad)
            ref = run(tokens)
            ref_other = run(other)
            warm = [
                ("chunked", {}, False, tokens, ref),
                ("cpf", cpf, False, tokens, ref),
                ("cpf_qkv", cpfq, False, tokens, ref),
                ("cpf_qkv_seqsplit", cpfq, True, tokens, ref),
                ("rep_w", rep, False, tokens, ref),
                ("rep_r", rep, False, tokens, ref),
                ("rep_other", rep, False, other, ref_other),
                ("rep_back_w", rep, False, tokens, ref),
                ("rep_back_r_all", rep | cpfq, True, tokens, ref),
            ]
            for name, env, ss, toks, r in warm:
                _set({}, False, pad)
                run(toks[:, :P])  # the cached prefix rows of this prompt, cold
                _set(env, ss, pad)
                o = run(toks, start=P)
                logger.info(f"CPFPCC replica valid={gen._cpf_rep_valid and gen._cpf_rep_valid[0]}")
                pcc, t1 = _cmp(f"warm n={n} {name}", r, o)
                if pcc < 0.98 or not t1:
                    bad.append(f"warm n={n} {name} pcc={pcc:.4f} top1={t1}")
    _set({}, SEQSPLIT_AT_LAUNCH, True)
    assert not bad, "\n".join(bad)
