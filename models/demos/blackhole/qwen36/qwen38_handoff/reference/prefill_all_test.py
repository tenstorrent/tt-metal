# Evidence-only helper: TT eager prefill_tp hidden states for 1024 tokens; per-position lm_head via one-hot select.
import sys

sys.path.insert(0, "/home/ttuser/atupe/tt-metal/.claude/worktrees/qwen38-optimizations")
import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.model_config import Mode


@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_prefill_all(mesh_device):
    O = "/home/ttuser/atupe/qwen38_work/runs/T3/"
    R = "models/tt_transformers/tests/reference_outputs/Qwen3.8-27B.refpt"
    toks = torch.load(R)["reference_tokens"][:, :1024]
    hf = torch.load(O + "kl/hf38_logp.pt")
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=2048 + 1024)
    model.reset_tp()
    T = 1024
    model._build_request_rope(toks, None)
    tok = ttnn.from_torch(
        toks.to(torch.int32), dtype=ttnn.uint32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    x = model.embd(tok)
    x = ttnn.reshape(x, (1, 1, T, x.shape[-1]))
    cos_t, sin_t = model._rope_tp_cos_sin_torch(0, T)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
    sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
    for layer in model.layers:
        x = layer.forward(x, cos=cos, sin=sin, mode="prefill", chunk_size=128, valid_len=T)
    am = []
    lps = []
    for p in range(511, 1023):
        sel = torch.zeros(1, 1, 1, T)
        sel[0, 0, 0, p] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=x.dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        xl = ttnn.matmul(sel_tt, x)
        ttnn.deallocate(sel_tt)
        xl = ttnn.to_memory_config(xl, ttnn.DRAM_MEMORY_CONFIG)
        xl = model.norm(xl, mode=Mode.PREFILL)
        lg = model._lm_head(xl)
        lt = (
            ttnn.to_torch(lg, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
            .reshape(-1)[: model.vocab_size]
            .float()
        )
        am.append(int(lt.argmax()))
        lps.append(torch.log_softmax(lt, -1))
    am = torch.tensor(am)
    torch.save({"argmax": am}, O + "bisect/prefill_tt38_argmax.pt")
    ref = hf["argmax"]
    agree = (am == ref).float().mean().item()
    refpt = torch.load(R)["top5_tokens"][511:1023, 0]
    print(f"PREFILL_ONLY top1 vs HF38 argmax: {100*agree:.2f}%  vs refpt: {100*(am==refpt).float().mean().item():.2f}%")
    L = torch.stack(lps)
    P = torch.log_softmax(hf["logp"].float(), -1)
    kl = (P.exp() * (P - L)).sum(-1)
    print(f"PREFILL_ONLY KL mean {kl.mean():.4f} med {kl.median():.4f} p99 {kl.quantile(0.99):.4f}")
    for b in range(4):
        s = slice(b * 128, (b + 1) * 128)
        print("block", b * 128, "dis", int((am[s] != ref[s]).sum()), "meanKL", float(kl[s].mean()))
