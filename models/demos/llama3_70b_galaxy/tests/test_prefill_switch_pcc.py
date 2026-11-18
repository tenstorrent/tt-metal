# Accuracy of the BH prefill switches on the FULL 80-layer model: prefill a real-text prompt (4096 and 2048 tokens) with
# the switches off (reference) and on, compare last-token logits (PCC, top-1, top-5 overlap). One process, one model.
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_prefill_switch_pcc.py -s --timeout=0
import glob
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama3_70b_galaxy.demo.text_demo import create_tt_model
from models.demos.llama3_70b_galaxy.tt.generator import Generator
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM

SWITCHES = [
    "LLAMA_BH_PREFILL_WO_RS",
    "LLAMA_BH_PREFILL_FF2_MINIMAL",
    "LLAMA_BH_PREFILL_MINIMAL_2K",
    "LLAMA_BH_PREFILL_MM_TUNED",
    "LLAMA_BH_PREFILL_FF13_BF8",
]
VARIANTS = {
    "ref_again": ({}, "HiFi4"),
    "all_lofi": ({k: "1" for k in SWITCHES} | {"LLAMA_BH_PREFILL_RS_WORKERS": "4"}, "LoFi"),
    "all_hifi2": ({k: "1" for k in SWITCHES} | {"LLAMA_BH_PREFILL_RS_WORKERS": "4"}, "HiFi2"),
    "all_hifi4": ({k: "1" for k in SWITCHES} | {"LLAMA_BH_PREFILL_RS_WORKERS": "4"}, "HiFi4"),
}


def _set(model, env, fid):
    for k in SWITCHES + ["LLAMA_BH_PREFILL_RS_WORKERS"]:
        os.environ.pop(k, None)
    os.environ.update(env)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, fid),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    for layer in model.layers:
        layer.attention.compute_kernel_config_sdpa = cfg


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
def test_prefill_switch_pcc(mesh_device, device_params):
    model_args, model, page_table, kv = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=1,
        optimizations=LlamaOptimizations.performance,
        max_seq_len=131072,
        num_layers=80,
        dummy_weights=False,
        page_params={"page_block_size": 64, "page_max_num_blocks": 2048},
        dtype=ttnn.bfloat8_b,
        use_paged_kv_cache=True,
    )
    tok = model_args.create_tokenizer()
    model_args.tokenizer = tok
    gen = Generator(model, model_args, mesh_device, tokenizer=tok)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = "\n\n".join(
        open(f, errors="ignore").read() for f in sorted(glob.glob(os.path.join(root, "**/*.md"), recursive=True))
    )
    text += "\n\n".join(open(f, errors="ignore").read() for f in sorted(glob.glob(os.path.join(root, "tt/*.py"))))
    ids = tok.encode(text) if hasattr(tok, "encode") else tok(text)["input_ids"]
    logger.info(f"PCCSW corpus tokens {len(ids)}")
    pt = page_table[:1]
    seqs = [int(s) for s in os.environ.get("PCCSW_SEQS", "4096,2048").split(",")]
    for seq in seqs:
        tokens = torch.tensor(ids[:seq]).unsqueeze(0)

        def run():
            out = gen.prefill_forward_text(
                tokens, page_table=pt, kv_cache=kv, prompt_lens=[seq], enable_trace=False, sampling_params=None
            )
            return out[0] if isinstance(out, tuple) else out

        _set(model, {}, "HiFi4")
        ref = run().float().reshape(-1)[: model_args.vocab_size]
        ref_top5 = set(torch.topk(ref, 5).indices.tolist())
        for name, (env, fid) in VARIANTS.items():
            _set(model, env, fid)
            o = run().float().reshape(-1)[: model_args.vocab_size]
            pcc = torch.corrcoef(torch.stack([ref, o]))[0, 1].item()
            t1 = int(torch.argmax(ref) == torch.argmax(o))
            ov = len(ref_top5 & set(torch.topk(o, 5).indices.tolist()))
            logger.info(
                f"PCCSW seq={seq} {name:10s} pcc={pcc:.5f} top1_match={t1} top5_overlap={ov}/5 "
                f"ref_top1={int(torch.argmax(ref))} top1={int(torch.argmax(o))}"
            )
    _set(model, {}, "HiFi4")
