# models/demos/blackhole/qwen3_5_9b/tests/unit/test_multidevice_blocks.py
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tt.rope import Qwen35RoPESetup

# Single-device test: default to the 9B checkpoint (the 27B needs a multi-device mesh for TP).
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2}]


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a - a.mean(), b - b.mean()]))[0, 1].item()


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_rope_cos_sin_shapes(device):
    args = Qwen35ModelArgs(mesh_device=device, max_batch_size=1, max_seq_len=2048)
    rope = Qwen35RoPESetup(device, args)
    pos = torch.arange(8).unsqueeze(0)
    cos, sin = rope.get_rot_mats(pos)
    assert cos.shape[-1] == args.rope_head_dim
    assert sin.shape[-1] == args.rope_head_dim
    ch, sh = rope.get_cos_sin_host(0)
    assert tuple(ch.shape) == (1, 1, args.rope_head_dim)


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_rmsnorm_pcc(device):
    from models.common.rmsnorm import RMSNorm
    from models.tt_transformers.tt.common import Mode

    args = Qwen35ModelArgs(mesh_device=device, max_batch_size=1, max_seq_len=2048)
    dim = args.dim
    w = torch.randn(dim, dtype=torch.float32)
    sd = {"norm.weight": w}
    norm = RMSNorm(
        device=device,
        dim=dim,
        state_dict=sd,
        weight_key="norm",
        weight_cache_path=None,
        weight_dtype=ttnn.bfloat16,
        add_unit_offset=True,
        eps=args.norm_eps,
    )
    x = torch.randn(1, 1, 32, dim, dtype=torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    ref = (x * torch.rsqrt(var + args.norm_eps)) * (w + 1.0)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(norm(x_tt, mode=Mode.DECODE))
    assert _pcc(ref, out) > 0.99, f"RMSNorm PCC too low: {_pcc(ref, out)}"


@run_for_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_embedding_pcc(device):
    from models.tt_transformers.tt.embedding import Embedding

    args = Qwen35ModelArgs(mesh_device=device, max_batch_size=1, max_seq_len=2048)
    args.dummy_weights = True  # random table below; lets Embedding skip the weight cache path
    vocab, dim = args.vocab_size, args.dim
    table = torch.randn(vocab, dim, dtype=torch.bfloat16)
    sd = {"tok_embeddings.weight": table}
    emb = Embedding(mesh_device=device, args=args, weight_cache_path=None, state_dict=sd, dtype=ttnn.bfloat16)
    ids = torch.tensor([[1, 5, 9, 13]], dtype=torch.int32)
    ref = torch.nn.functional.embedding(ids.long(), table.float())
    ids_tt = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.to_torch(emb(ids_tt)).float()[..., :dim].reshape(ref.shape)
    assert _pcc(ref, out) > 0.99
