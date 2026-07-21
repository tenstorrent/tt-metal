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


def test_encoder_parallel_config_from_tuples():
    pc = EncoderParallelConfig.from_tuples(tp=(4, 1), sp=(4, 0), cfg=(2, 1))
    assert pc.tensor_parallel == ParallelFactor(4, 1)
    assert pc.sequence_parallel == ParallelFactor(4, 0)
    assert pc.cfg_parallel == ParallelFactor(2, 1)

    # back-compat: from_tuple sets only tensor_parallel, leaves sp/cfg None
    legacy = EncoderParallelConfig.from_tuple((8, 1))
    assert legacy.tensor_parallel == ParallelFactor(8, 1)
    assert legacy.sequence_parallel is None
    assert legacy.cfg_parallel is None


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
        try:
            assert_quality(r, tt_tensor.to_torch(d), pcc=0.99)
        except Exception as e:
            raise Exception(f"state {i}: {e}") from e


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_smollm3_encode_contract(*, mesh_device):
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
    ref_prompt = torch.cat([ref.hidden_states[-1], ref.hidden_states[-2]], dim=-1).float()

    cfg = SmolLM3Config.from_hf_config(hf.config)
    cfg.num_hidden_layers = n_layers
    cfg.no_rope_layers = cfg.no_rope_layers[:n_layers]
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pc = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    enc = SmolLM3TextEncoder(cfg, device=mesh_device, parallel_config=pc, ccl_manager=ccl)
    enc.load_torch_state_dict(hf.model.state_dict())

    cos, sin = enc.create_rope_tensors(1, seq)
    tt_ids = tt_tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    prompt_embeds, hs = enc.encode(
        tt_ids,
        attention_mask=None,
        pos_embeds=(tt_tensor.from_torch(cos, device=mesh_device), tt_tensor.from_torch(sin, device=mesh_device)),
    )
    out = tt_tensor.to_torch(prompt_embeds)
    assert out.shape[-1] == 2 * cfg.hidden_size
    assert_quality(ref_prompt, out, pcc=0.99)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_smollm3_encoder_masked(*, mesh_device):
    from models.tt_dit.encoders.smollm3.config import SmolLM3Config
    from models.tt_dit.encoders.smollm3.model_smollm3 import SmolLM3TextEncoder

    torch.manual_seed(0)
    n_layers = int(os.environ.get("N_LAYERS", "6"))
    seq = 128
    n_real = 100  # right-padding: real tokens first, padding after

    hf = _load_hf_smollm3()
    hf.model.layers = hf.model.layers[:n_layers]
    hf.config.num_hidden_layers = n_layers

    tokens = torch.randint(1, hf.config.vocab_size, (1, seq))
    attn = torch.zeros(1, seq, dtype=torch.long)
    attn[:, :n_real] = 1

    # HF reference with attention_mask
    with torch.no_grad():
        ref = hf.model(input_ids=tokens, attention_mask=attn, output_hidden_states=True)
    ref_hs = [h.float() for h in ref.hidden_states]

    cfg = SmolLM3Config.from_hf_config(hf.config)
    cfg.num_hidden_layers = n_layers
    cfg.no_rope_layers = cfg.no_rope_layers[:n_layers]

    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pc = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    enc = SmolLM3TextEncoder(cfg, device=mesh_device, parallel_config=pc, ccl_manager=ccl)
    enc.load_torch_state_dict(hf.model.state_dict())

    cos, sin = enc.create_rope_tensors(1, seq)

    tt_ids = tt_tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    # attention_mask: numeric (B, seq) float tensor — prepare_attention_bias does (mask - 1.0) * inf
    tt_mask = tt_tensor.from_torch(attn.float(), device=mesh_device)
    tt_cos = tt_tensor.from_torch(cos, device=mesh_device)
    tt_sin = tt_tensor.from_torch(sin, device=mesh_device)

    hs = enc.forward(tt_ids, attention_mask=tt_mask, pos_embeds=(tt_cos, tt_sin))

    assert len(hs) == len(ref_hs), f"got {len(hs)} states, expected {len(ref_hs)}"
    for i, (r, d) in enumerate(zip(ref_hs, hs)):
        # Compare only real (non-padding) positions; padding positions are undefined.
        try:
            assert_quality(r[:, :n_real, :], tt_tensor.to_torch(d)[:, :n_real, :], pcc=0.99)
        except Exception as e:
            raise Exception(f"state {i}: {e}") from e


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 8192}],
    indirect=["device_params"],
)
@pytest.mark.parametrize("seq", [128, 2048])
def test_smollm3_encoder_full_mesh(*, mesh_device, seq):
    from models.tt_dit.encoders.smollm3.config import SmolLM3Config
    from models.tt_dit.encoders.smollm3.model_smollm3 import SmolLM3TextEncoder

    torch.manual_seed(0)
    tp_axis = 1
    hf = _load_hf_smollm3()
    tokens = torch.randint(0, hf.config.vocab_size, (1, seq))
    with torch.no_grad():
        ref = hf.model(input_ids=tokens, output_hidden_states=True)
    ref_prompt = torch.cat([ref.hidden_states[-1], ref.hidden_states[-2]], dim=-1).float()

    cfg = SmolLM3Config.from_hf_config(hf.config)
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pc = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis))
    enc = SmolLM3TextEncoder(cfg, device=mesh_device, parallel_config=pc, ccl_manager=ccl)
    enc.load_torch_state_dict(hf.model.state_dict())

    cos, sin = enc.create_rope_tensors(1, seq)
    tt_ids = tt_tensor.from_torch(tokens, device=mesh_device, dtype=ttnn.uint32)
    prompt_embeds, _ = enc.encode(
        tt_ids,
        attention_mask=None,
        pos_embeds=(tt_tensor.from_torch(cos, device=mesh_device), tt_tensor.from_torch(sin, device=mesh_device)),
    )
    assert_quality(ref_prompt, tt_tensor.to_torch(prompt_embeds), pcc=0.99)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 8192}],
    indirect=["device_params"],
)
@pytest.mark.parametrize("seq", [512])  # divisible by max sp_factor(8)*32 = 256 (and by 4*32)
@pytest.mark.parametrize("sp_axis", [0, 1])  # 0 -> SP=4/TP=8 ; 1 -> SP=8/TP=4 (axis-swapped)
def test_smollm3_encoder_sp(*, mesh_device, seq, sp_axis):
    """Sequence-parallel (all-gather K/V) encoder: seq sharded on sp_axis, tp on the other axis.

    Covers both layouts on the 4x8 mesh: sp_axis=0 (SP=4 x TP=8) and sp_axis=1 (SP=8 x TP=4).
    Inputs (input_ids, rope cos/sin) are sharded along the seq dim over sp_axis; outputs are
    gathered back over the same axis. PCC vs. HF proves the RoPE-shard-offset + rectangular
    causal-bias math for either axis assignment.
    """
    from models.tt_dit.encoders.smollm3.model_smollm3 import SmolLM3TextEncoder

    torch.manual_seed(0)
    tp_axis = 1 - sp_axis
    sp_factor = mesh_device.shape[sp_axis]
    tp_factor = mesh_device.shape[tp_axis]
    hf = _load_hf_smollm3()
    tokens = torch.randint(0, hf.config.vocab_size, (1, seq))
    with torch.no_grad():
        ref = hf.model(input_ids=tokens, output_hidden_states=True)
    ref_prompt = torch.cat([ref.hidden_states[-1], ref.hidden_states[-2]], dim=-1).float()

    cfg = SmolLM3Config.from_hf_config(hf.config)
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pc = EncoderParallelConfig.from_tuples(tp=(tp_factor, tp_axis), sp=(sp_factor, sp_axis))
    enc = SmolLM3TextEncoder(cfg, device=mesh_device, parallel_config=pc, ccl_manager=ccl)
    enc.load_torch_state_dict(hf.model.state_dict())

    cos, sin = enc.create_rope_tensors(1, seq)  # (1,1,seq,head_dim); shard seq (axis 2) on sp_axis
    tt_ids = tt_tensor.from_torch(
        tokens, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=[None, sp_axis]
    )
    tt_cos = tt_tensor.from_torch(cos, device=mesh_device, mesh_axes=[None, None, sp_axis, None])
    tt_sin = tt_tensor.from_torch(sin, device=mesh_device, mesh_axes=[None, None, sp_axis, None])
    prompt_embeds, _ = enc.encode(tt_ids, attention_mask=None, pos_embeds=(tt_cos, tt_sin))
    out = tt_tensor.to_torch(prompt_embeds, mesh_axes=[None, sp_axis, None], composer_device=mesh_device)
    assert_quality(ref_prompt, out, pcc=0.99, relative_rmse=0.2)


def test_fibo_wrapper_bucket_pick(*, expect_error):
    from models.tt_dit.pipelines.bria_fibo.text_encoder import pick_bucket

    assert pick_bucket(10, (1024,), sp_factor=4) == 1024
    assert pick_bucket(1024, (1024, 2048), sp_factor=4) == 1024
    assert pick_bucket(1025, (1024, 2048), sp_factor=4) == 2048
    with expect_error(ValueError, "exceeds all buckets"):
        pick_bucket(3000, (1024, 2048), sp_factor=4)
    with expect_error(ValueError, "not divisible"):
        pick_bucket(10, (1000,), sp_factor=4)  # 1000 % (4*32) != 0


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 8192}],
    indirect=["device_params"],
)
def test_smollm3_sp_bias_cached(*, mesh_device):
    """The SP causal bias is built once per local seq length and reused across forwards."""
    from models.tt_dit.encoders.smollm3.model_smollm3 import SmolLM3TextEncoder

    torch.manual_seed(0)
    sp_axis, tp_axis, seq = 1, 0, 512
    sp_factor, tp_factor = mesh_device.shape[sp_axis], mesh_device.shape[tp_axis]
    hf = _load_hf_smollm3()
    cfg = SmolLM3Config.from_hf_config(hf.config)
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pc = EncoderParallelConfig.from_tuples(tp=(tp_factor, tp_axis), sp=(sp_factor, sp_axis))
    enc = SmolLM3TextEncoder(cfg, device=mesh_device, parallel_config=pc, ccl_manager=ccl)
    enc.load_torch_state_dict(hf.model.state_dict())

    tokens = torch.randint(0, hf.config.vocab_size, (1, seq))
    cos, sin = enc.create_rope_tensors(1, seq)
    tt_ids = tt_tensor.from_torch(
        tokens, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=[None, sp_axis]
    )
    tt_cos = tt_tensor.from_torch(cos, device=mesh_device, mesh_axes=[None, None, sp_axis, None])
    tt_sin = tt_tensor.from_torch(sin, device=mesh_device, mesh_axes=[None, None, sp_axis, None])

    enc.encode(tt_ids, attention_mask=None, pos_embeds=(tt_cos, tt_sin))
    assert len(enc._sp_bias_cache) == 1
    first = next(iter(enc._sp_bias_cache.values()))
    enc.encode(tt_ids, attention_mask=None, pos_embeds=(tt_cos, tt_sin))
    assert len(enc._sp_bias_cache) == 1
    assert next(iter(enc._sp_bias_cache.values())) is first  # same tensor object reused


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 8192}],
    indirect=["device_params"],
)
def test_fibo_wrapper_encode(*, mesh_device):
    """Wrapper ``encode_prompt`` fidelity: the stacked-and-split readback matches HF on the JSON prompt.

    Exercises the whole-mesh SP x TP wrapper end to end (tokenize -> padded 1024 bucket -> device forward
    -> SP-sharded readback -> host-derived ``prompt_embeds``). Runs the full prompt and a truncated half
    through the shared 1024 bucket (different real lengths, same bucket), then checks the full-prompt
    output against HF. ``len(hidden) == len(ref.hidden_states)`` (the transformer-block count follows from
    ``build_text_encoder_layers``).
    """
    from pathlib import Path

    from huggingface_hub import snapshot_download

    from models.tt_dit.pipelines.bria_fibo.text_encoder import SmolLM3TextEncoderWrapper

    sp_axis, tp_axis = 1, 0
    pc = EncoderParallelConfig.from_tuples(
        tp=(mesh_device.shape[tp_axis], tp_axis), sp=(mesh_device.shape[sp_axis], sp_axis)
    )
    try:
        ckpt = snapshot_download(FIBO_PATH, local_files_only=True)
    except Exception as e:
        pytest.skip(f"FIBO unavailable: {e}")

    json_path = Path(__file__).resolve().parents[2] / "models" / "bria_fibo" / "fibo_vlm_prompt.json"
    if not json_path.is_file():
        pytest.skip(f"JSON prompt fixture missing: {json_path}")
    json_prompt = json_path.read_text().strip()

    ccl = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Linear)
    wrapper = SmolLM3TextEncoderWrapper(
        ckpt, device=mesh_device, ccl_manager=ccl, parallel_config=pc, pad_buckets=(1024,)
    )

    # Full prompt + a truncated half exercise different real lengths through the shared 1024 bucket.
    outs = [wrapper.encode_prompt(p) for p in (json_prompt, json_prompt[: len(json_prompt) // 2])]
    for embeds, hidden in outs:
        assert embeds.ndim == 3 and embeds.shape[0] == 1
        assert len(hidden) > 0

    # The stacked-and-split readback (+ host-derived prompt_embeds) must match HF on the realistic prompt.
    hf = _load_hf_smollm3()
    ids = wrapper.tokenizer([json_prompt], add_special_tokens=True, return_tensors="pt").input_ids
    with torch.no_grad():
        ref = hf.model(input_ids=ids, output_hidden_states=True)
    ref_embeds = torch.cat([ref.hidden_states[-1], ref.hidden_states[-2]], dim=-1).float()
    embeds, hidden = outs[0]
    assert len(hidden) == len(ref.hidden_states)
    assert list(embeds.shape) == list(ref_embeds.shape)
    assert_quality(ref_embeds, embeds.float(), pcc=0.99, relative_rmse=0.2)
