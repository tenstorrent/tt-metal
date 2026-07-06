# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

import ttnn
from models.tt_dit.encoders.smollm3.config import SmolLM3Config
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor as tt_tensor
from models.tt_dit.utils.check import assert_quality


def test_smollm3_config_defaults():
    c = SmolLM3Config()
    assert c.hidden_size == 2048
    assert c.num_attention_heads == 16
    assert c.num_key_value_heads == 4
    assert c.head_dim == 128
    assert c.num_hidden_layers == 36
    assert c.intermediate_size == 11008
    assert c.rope_theta == 5000000.0
    assert c.rms_norm_eps == 1e-6
    assert c.vocab_size == 128256
    assert c.attention_bias is False
    # NoPE on every 4th layer (0-indexed 3,7,...,35); 1 = apply rope, 0 = NoPE
    assert len(c.no_rope_layers) == 36
    assert c.no_rope_layers[0] == 1 and c.no_rope_layers[3] == 0 and c.no_rope_layers[7] == 0
    assert sum(c.no_rope_layers) == 27  # 36 - 9 NoPE layers


def test_smollm3_rope_matches_hf():
    from models.tt_dit.encoders.smollm3.model_smollm3 import create_rope_tensors

    head_dim, rope_theta, batch, seq = 128, 5000000.0, 2, 40
    cos, sin = create_rope_tensors(batch, seq, head_dim, rope_theta)
    assert cos.shape == (batch, 1, seq, head_dim)
    assert sin.shape == (batch, 1, seq, head_dim)

    # HF reference: inv_freq then emb=cat(freqs,freqs); cos/sin over (seq, head_dim)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    pos = torch.arange(seq).float()
    freqs = torch.outer(pos, inv_freq)  # (seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq, head_dim)
    ref_cos, ref_sin = emb.cos(), emb.sin()

    torch.testing.assert_close(cos[0, 0], ref_cos, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sin[0, 0], ref_sin, atol=1e-5, rtol=1e-5)


FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")


def _load_hf_smollm3():
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(FIBO_PATH, subfolder="text_encoder", torch_dtype=torch.float32)
    except Exception as e:  # gated / offline
        pytest.skip(f"FIBO text_encoder unavailable: {e}")
    return model.eval()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_smollm3_mlp(*, mesh_device):
    from models.tt_dit.encoders.smollm3.model_smollm3 import SmolLM3Context, SmolLM3Mlp

    torch.manual_seed(0)
    hf = _load_hf_smollm3()
    hf_mlp = hf.model.layers[0].mlp
    cfg = hf.config
    ctx = SmolLM3Context(device=mesh_device, tp_axis=None, ccl_manager=None)

    mlp = SmolLM3Mlp(cfg.hidden_size, cfg.intermediate_size, cfg.hidden_act, ctx)
    mlp.load_torch_state_dict(hf_mlp.state_dict())

    x = torch.randn(1, 128, cfg.hidden_size)
    with torch.no_grad():
        ref = hf_mlp(x)
    tt_x = tt_tensor.from_torch(x, device=mesh_device)
    tt_out = mlp.forward(tt_x)
    assert_quality(ref, tt_tensor.to_torch(tt_out), pcc=0.99)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
@pytest.mark.parametrize("use_rope", [pytest.param(True, id="rope"), pytest.param(False, id="nope")])
def test_smollm3_attention(*, mesh_device, use_rope):
    from models.tt_dit.encoders.smollm3.model_smollm3 import SmolLM3Attention, SmolLM3Context, create_rope_tensors

    torch.manual_seed(0)
    hf = _load_hf_smollm3()
    cfg = hf.config
    seq = 128

    # Pick a reference layer whose HF use_rope matches, then force it to be safe.
    hf_attn = hf.model.layers[0].self_attn
    hf_attn.use_rope = use_rope

    ctx = SmolLM3Context(device=mesh_device, tp_axis=None, ccl_manager=None)
    attn = SmolLM3Attention(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        use_rope=use_rope,
        ctx=ctx,
    )
    attn.load_torch_state_dict(hf_attn.state_dict())

    head_dim = cfg.hidden_size // cfg.num_attention_heads
    rope_theta = cfg.rope_parameters["rope_theta"]

    x = torch.randn(1, seq, cfg.hidden_size)
    cos, sin = create_rope_tensors(1, seq, head_dim, rope_theta)

    # HF reference: pure-causal (all real tokens), so device is_causal path matches.
    with torch.no_grad():
        ref, _ = hf_attn(
            x,
            position_embeddings=(cos[:, 0], sin[:, 0]),  # HF expects (B, seq, head_dim)
            attention_mask=None,
        )

    tt_x = tt_tensor.from_torch(x, device=mesh_device)
    tt_cos = tt_tensor.from_torch(cos, device=mesh_device)
    tt_sin = tt_tensor.from_torch(sin, device=mesh_device)
    tt_out = attn.forward(tt_x, attention_bias=None, pos_embeds=(tt_cos, tt_sin))
    assert_quality(ref, tt_tensor.to_torch(tt_out), pcc=0.99, relative_rmse=0.2)


def test_smollm3_config_from_hf():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoConfig

    try:
        hf = AutoConfig.from_pretrained(os.environ.get("FIBO_PATH", "briaai/FIBO"), subfolder="text_encoder")
    except Exception as e:
        pytest.skip(f"FIBO config unavailable: {e}")
    cfg = SmolLM3Config.from_hf_config(hf)
    assert cfg.rope_theta == 5000000.0
    assert cfg.head_dim == 128
    assert cfg.num_hidden_layers == 36
    assert cfg.hidden_size == 2048
    assert len(cfg.no_rope_layers) == 36 and sum(cfg.no_rope_layers) == 27
    assert cfg.attention_bias is False


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_smollm3_decoder_layer(*, mesh_device):
    from models.tt_dit.encoders.smollm3.config import SmolLM3Config
    from models.tt_dit.encoders.smollm3.model_smollm3 import SmolLM3Context, SmolLM3DecoderLayer, create_rope_tensors

    torch.manual_seed(0)
    hf = _load_hf_smollm3()
    cfg = hf.config
    seq = 128
    hf_layer = hf.model.layers[0]  # layer 0 is a RoPE layer

    sm_cfg = SmolLM3Config.from_hf_config(cfg)

    ctx = SmolLM3Context(device=mesh_device, tp_axis=None, ccl_manager=None)

    # layer 0 has no_rope_layers[0] == 1 → use_rope=True
    use_rope = bool(sm_cfg.no_rope_layers[0])

    layer = SmolLM3DecoderLayer(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        hidden_act=cfg.hidden_act,
        rms_norm_eps=cfg.rms_norm_eps,
        use_rope=use_rope,
        ctx=ctx,
    )
    layer.load_torch_state_dict(hf_layer.state_dict())

    x = torch.randn(1, seq, cfg.hidden_size)
    cos, sin = create_rope_tensors(1, seq, sm_cfg.head_dim, sm_cfg.rope_theta)

    # HF reference: attention_mask=None → SDPA uses is_causal=True (module.is_causal=True),
    # matching the TT layer's is_causal=attention_bias is None path.
    with torch.no_grad():
        ref = hf_layer(
            x,
            position_embeddings=(cos[:, 0], sin[:, 0]),  # HF expects (B, seq, head_dim)
            attention_mask=None,
        )
        ref = ref[0] if isinstance(ref, tuple) else ref

    tt_x = tt_tensor.from_torch(x, device=mesh_device)
    tt_cos = tt_tensor.from_torch(cos, device=mesh_device)
    tt_sin = tt_tensor.from_torch(sin, device=mesh_device)
    tt_out = layer.forward(
        tt_x,
        attention_bias=None,
        pos_embeds=(tt_cos, tt_sin),
    )
    assert_quality(ref, tt_tensor.to_torch(tt_out), pcc=0.99, relative_rmse=0.2)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_smollm3_encoder_all_layers(*, mesh_device):
    from models.tt_dit.encoders.smollm3.config import SmolLM3Config
    from models.tt_dit.encoders.smollm3.model_smollm3 import SmolLM3TextEncoder

    torch.manual_seed(0)
    n_layers = int(os.environ.get("N_LAYERS", "6"))
    seq = 128
    hf = _load_hf_smollm3()
    hf.model.layers = hf.model.layers[:n_layers]
    hf.config.num_hidden_layers = n_layers

    tokens = torch.randint(0, hf.config.vocab_size, (1, seq))
    with torch.no_grad():
        ref = hf.model(input_ids=tokens, output_hidden_states=True)
    ref_hs = [h.float() for h in ref.hidden_states]  # length n_layers + 1

    cfg = SmolLM3Config.from_hf_config(hf.config)
    cfg.num_hidden_layers = n_layers
    cfg.no_rope_layers = cfg.no_rope_layers[:n_layers]

    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pc = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    enc = SmolLM3TextEncoder(cfg, device=mesh_device, parallel_config=pc, ccl_manager=ccl)
    enc.load_torch_state_dict(hf.model.state_dict())

    cos, sin = enc.create_rope_tensors(1, seq)
    tt_ids = tt_tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    hs = enc.forward(
        tt_ids,
        attention_mask=None,
        pos_embeds=(tt_tensor.from_torch(cos, device=mesh_device), tt_tensor.from_torch(sin, device=mesh_device)),
    )
    assert len(hs) == len(ref_hs), f"got {len(hs)} states, expected {len(ref_hs)}"
    for i, (r, d) in enumerate(zip(ref_hs, hs)):
        assert_quality(r, tt_tensor.to_torch(d), pcc=0.99)
