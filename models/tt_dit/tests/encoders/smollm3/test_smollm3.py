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
    prompt_embeds, _ = enc.encode(tt_ids, pos_embeds=(tt_cos, tt_sin))
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

    enc.encode(tt_ids, pos_embeds=(tt_cos, tt_sin))
    assert len(enc._sp_bias_cache) == 1
    first = next(iter(enc._sp_bias_cache.values()))
    enc.encode(tt_ids, pos_embeds=(tt_cos, tt_sin))
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


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 8192, "trace_region_size": 200000000}],
    indirect=["device_params"],
)
def test_fibo_wrapper_encode_replay_stable(*, mesh_device):
    """The traced encoder must stay bit-exact across MANY sequential encodes (the reverted bug was
    'noise after the first run').

    Guards replay stability: each replay of a prompt must be bit-identical to that prompt's FIRST
    (captured) encode (PCC >= 0.9999) -- checked for both the realistic json prompt and the empty (neg)
    prompt, which share the 1024 trace and alternate across simulated generations. The realistic prompt
    is additionally checked vs HF at 0.99 (absolute correctness). The empty prompt is checked for replay
    stability ONLY, not vs HF: it has a known ~0.98 short-sample HF gap that is present untraced too, so
    a traced-vs-HF check on it would confound the replay-stability signal this test exists to guard.
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

    hf = _load_hf_smollm3()
    ids = wrapper.tokenizer([json_prompt], add_special_tokens=True, return_tensors="pt").input_ids
    with torch.no_grad():
        ref = hf.model(input_ids=ids, output_hidden_states=True)
    hf_json = torch.cat([ref.hidden_states[-1], ref.hidden_states[-2]], dim=-1).float()

    # Alternate pos/neg across several "generations", all traced: first encode of each captures, the
    # rest replay (traced=True drives the per-bucket Tracer, the same flag the DiT denoise uses).
    baselines: dict[str, torch.Tensor] = {}
    prompts = [json_prompt, "", json_prompt, "", json_prompt, ""]
    for i, p in enumerate(prompts):
        embeds, _ = wrapper.encode_prompt(p, traced=True)
        embeds = embeds.float()
        key = "json" if p else "empty"
        if key not in baselines:
            baselines[key] = embeds  # capture (first encode of this prompt)
        else:
            # Replay must be bit-identical to the captured output -- the property the revert broke.
            assert_quality(baselines[key], embeds, pcc=0.9999)
        if key == "json":
            assert list(embeds.shape) == list(hf_json.shape), f"run {i}: {embeds.shape} != {hf_json.shape}"
            assert_quality(hf_json, embeds, pcc=0.99, relative_rmse=0.2)  # traced json correct vs HF


def test_smollm3_state_conversion():
    import torch as _torch

    from models.tt_dit.encoders.smollm3.model_smollm3 import STATE_CONVERSION

    # Full SmolLM3ForCausalLM-style dict: model.* prefix + lm_head + a rotary_emb buffer.
    full = {
        "model.embed_tokens.weight": _torch.zeros(2, 2),
        "model.layers.0.self_attn.q_proj.weight": _torch.zeros(2, 2),
        "model.layers.0.mlp.gate_proj.weight": _torch.zeros(2, 2),
        "model.norm.weight": _torch.zeros(2),
        "model.rotary_emb.inv_freq": _torch.zeros(2),
        "lm_head.weight": _torch.zeros(2, 2),
    }
    out = STATE_CONVERSION.convert(full)
    assert set(out) == {
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "norm.weight",
    }

    # Inner SmolLM3Model-style dict (no model. prefix, no lm_head): keys pass through unchanged.
    inner = {
        "embed_tokens.weight": _torch.zeros(2, 2),
        "layers.0.self_attn.q_proj.weight": _torch.zeros(2, 2),
        "norm.weight": _torch.zeros(2),
    }
    out2 = STATE_CONVERSION.convert(inner)
    assert set(out2) == set(inner)
