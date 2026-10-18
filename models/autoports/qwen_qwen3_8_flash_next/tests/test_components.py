"""Real target-shaped weight checks for residual mixing and active MoE experts."""
import os
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual, Qwen4ExpTextSparseMoeBlock

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.weight_stats import DEFAULT_CHECKPOINT
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.hyperconnection import HyperConnection
from models.autoports.qwen_qwen3_8_flash_next.tt.moe import MoE


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", DEFAULT_CHECKPOINT))


def config():
    return AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config


def layer_state(layer_idx):
    return CheckpointState(CHECKPOINT, f"model.language_model.layers.{layer_idx}.")


def test_hyper_real():
    torch.set_num_threads(8)
    c = config()
    state = layer_state(0)
    prefix = "attn_hyper_connection"
    hf = Qwen4ExpTextGatedResidual(c).to(torch.bfloat16).eval()
    hf.load_state_dict({k[len(prefix) + 1 :]: state[k] for k in state if k.startswith(prefix + ".")})
    torch.manual_seed(0)
    x = torch.randn(1, 32, 10240, dtype=torch.bfloat16) * 0.1
    with torch.no_grad():
        expected, _, injection = hf(x)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        tt = HyperConnection.from_state_dict(state, prefix=prefix, hf_config=c, mesh_device=mesh)
        tx = ttnn.from_torch(x.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        out, inj = tt.read(tx)
        got = ttnn.to_torch(out).reshape(expected.shape)
        pinj = pcc(injection, ttnn.to_torch(inj).reshape(injection.shape))
        score = pcc(expected, got)
        print("HYPER_PCC", score, "INJECTION_PCC", pinj, flush=True)
        assert min(score, pinj) >= 0.995
        actual = ttnn.to_torch(tt.write(tx, out, inj)).reshape(x.shape)
        ref = x + (expected.unsqueeze(-2) * injection.unsqueeze(-1)).flatten(-2)
        assert pcc(actual, ref) >= 0.995
    finally:
        ttnn.close_mesh_device(mesh)


def test_moe_real():
    torch.set_num_threads(8)
    c = config()
    state = layer_state(0)
    with torch.device("meta"):
        hf = Qwen4ExpTextSparseMoeBlock(c)
    hf.load_state_dict({k[4:]: state[k] for k in state if k.startswith("mlp.")}, assign=True)
    hf.eval()
    torch.manual_seed(3)
    x = torch.randn(1, 32, 2560, dtype=torch.bfloat16) * 0.1
    with torch.no_grad():
        expected = hf(x)
    print("HF_MOE_DONE", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        model = MoE.from_state_dict(state, hf_config=c, mesh_device=mesh)
        print("TT_MOE_LOADED", flush=True)
        tx = ttnn.from_torch(x.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        out = model(tx)
        actual = ttnn.to_torch(out).reshape(expected.shape)
        score = pcc(expected, actual)
        print("MOE_PCC", score, flush=True)
        assert score >= 0.995
    finally:
        ttnn.close_mesh_device(mesh)
