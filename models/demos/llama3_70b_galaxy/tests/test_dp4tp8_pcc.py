# DP4 x TP8 prefill (tt/prefill_dp4tp8.py) vs the TP32 prefill, real weights and real text, through the generator.
# Four prompts on users 0 / 8 / 16 / 24 (one per mesh column) are prefilled twice from cold:
#   ref: LLAMA_BH_PREFILL_DP4_OFF=1 (today's TP32 path, one user after another)
#   dp4: the DP4 x TP8 path (all four at once), writing into a different set of KV blocks
# Checks per user: last-token logits PCC / top-1 / top-5 overlap, and the KV cache rows it wrote (layers 0 and last,
# mesh rows 0 and 7 = kv heads 0 and 7, K and V) against the reference. Prints both prefill times.
# Launch with the server env (source sla_scripts/bh_tp32/env.sh) plus LLAMA_BH_PREFILL_DP4=1:
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_dp4tp8_pcc.py -s --timeout=0
# env: DP4PCC_LAYERS (80), DP4PCC_LENS (4000,3500,4096,2900)
import glob
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama3_70b_galaxy.demo.text_demo import create_tt_model
from models.demos.llama3_70b_galaxy.tt.generator import Generator
from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM

BS = 64
# DP4PCC_USERS / DP4PCC_LENS: one entry per prompt
USERS = [int(u) for u in (os.environ.get("DP4PCC_USERS") or "0,8,16,24").split(",")]


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


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
def test_dp4tp8_pcc(mesh_device, device_params):
    assert os.environ.get("LLAMA_BH_PREFILL_DP4") == "1", "launch with LLAMA_BH_PREFILL_DP4=1"
    lens = [int(x) for x in (os.environ.get("DP4PCC_LENS") or "4000,3500,4096,2900").split(",")]
    model_args, model, _, kv = create_tt_model(
        mesh_device,
        instruct=True,
        max_batch_size=32,
        optimizations=LlamaOptimizations.performance,
        max_seq_len=131072,
        num_layers=int(os.environ.get("DP4PCC_LAYERS") or "80"),
        dummy_weights=False,
        page_params={"page_block_size": BS, "page_max_num_blocks": 2048},
        dtype=ttnn.bfloat8_b,
        use_paged_kv_cache=True,
    )
    assert model.dp4 is not None
    tok = model_args.create_tokenizer()
    model_args.tokenizer = tok
    gen = Generator(model, model_args, mesh_device, tokenizer=tok)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = "\n\n".join(
        open(f, errors="ignore").read() for f in sorted(glob.glob(os.path.join(root, "**/*.md"), recursive=True))
    )
    text += "\n\n".join(open(f, errors="ignore").read() for f in sorted(glob.glob(os.path.join(root, "tt/*.py"))))
    ids = tok.encode(text) if hasattr(tok, "encode") else tok(text)["input_ids"]
    maxL = max(lens)
    N = len(lens)
    assert N == len(USERS), f"{N} lengths for {len(USERS)} users"
    tokens = torch.zeros(N, maxL, dtype=torch.long)
    for k, L in enumerate(lens):
        o = (k * 20000) % max(
            1, len(ids) - L
        )  # same prompts as before for the first 4 users when the text is long enough
        tokens[k, :L] = torch.tensor(ids[o : o + L])
    nb = (maxL + BS - 1) // BS + 2
    pt_ref = torch.stack([torch.arange(1 + k * 64, 1 + k * 64 + nb, dtype=torch.int32) for k in range(N)])
    pt_dp4 = pt_ref + 1024
    kv_layers = kv[0] if isinstance(kv[0], (list, tuple)) and isinstance(kv[0][0], (list, tuple)) else kv

    def run(pt, off):
        os.environ["LLAMA_BH_PREFILL_DP4_OFF"] = "1" if off else "0"
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = gen.prefill_forward_text(
            tokens,
            page_table=pt,
            kv_cache=kv,
            prompt_lens=lens,
            enable_trace=False,
            sampling_params=None,
            empty_slots=USERS,
        )
        ttnn.synchronize_device(mesh_device)
        return out.float().reshape(N, -1)[:, : model_args.vocab_size], time.perf_counter() - t0

    run(pt_ref, True)  # compile both paths once (the DP4 warmup inside the first call also captures its traces)
    run(pt_dp4, False)
    run(pt_dp4, False)
    ref, t_ref = run(pt_ref, True)
    if os.environ.get("DP4PCC_DEVPROF") == "1":
        # device profiler: flush everything so far, so its buffers hold exactly the next DP4 forward
        ttnn.ReadDeviceProfiler(mesh_device)
    dp4, t_dp4 = run(pt_dp4, False)
    if os.environ.get("DP4PCC_DEVPROF") == "1":
        ttnn.ReadDeviceProfiler(mesh_device)
    logger.info(f"DP4PCC time: TP32 {t_ref:.3f} s, DP4xTP8 {t_dp4:.3f} s for 4 prompts {lens} ({t_ref / t_dp4:.2f}x)")
    bad = []
    for k in range(N):
        p = _pcc(ref[k], dp4[k])
        t1 = int(torch.argmax(ref[k]) == torch.argmax(dp4[k]))
        ov = len(set(torch.topk(ref[k], 5).indices.tolist()) & set(torch.topk(dp4[k], 5).indices.tolist()))
        logger.info(f"DP4PCC user {USERS[k]} len {lens[k]}: logits pcc={p:.5f} top1_match={t1} top5={ov}/5")
        logger.info(
            f"DP4PCC user {USERS[k]} stats ref max {ref[k].max():.2f} std {ref[k].std():.3f} argmax {int(ref[k].argmax())} | dp4 max {dp4[k].max():.2f} std {dp4[k].std():.3f} argmax {int(dp4[k].argmax())} nan {int(torch.isnan(dp4[k]).sum())}"
        )
        if p < float(os.environ.get("DP4PCC_MIN") or "0.98") or not t1:
            bad.append(f"user {USERS[k]} logits pcc {p:.4f} top1 {t1}")
    n_layers = len(kv_layers)
    for li in (0, n_layers - 1):
        for kvi, name in ((0, "K"), (1, "V")):
            devs = ttnn.get_device_tensors(kv_layers[li][kvi])
            for r in (0, 7):
                for k in range(N):
                    c = USERS[k] // 8
                    full = ttnn.to_torch(devs[r * 4 + c]).float()  # [blocks, 1, BS, D]
                    nblk = (lens[k] + BS - 1) // BS
                    a = full[pt_ref[k, :nblk].long()].reshape(-1, full.shape[-1])[: lens[k]]
                    b = full[pt_dp4[k, :nblk].long()].reshape(-1, full.shape[-1])[: lens[k]]
                    p = _pcc(a, b)
                    logger.info(
                        f"DP4PCC KV layer {li} {name} row {r} user {USERS[k]}: pcc={p:.5f} max_abs={(a - b).abs().max():.3f}"
                    )
                    if p < float(os.environ.get("DP4PCC_KVMIN") or "0.98"):
                        bad.append(f"KV L{li} {name} r{r} u{USERS[k]} pcc {p:.4f}")
    assert not bad, "\n".join(bad)
