# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# MiniMax-H3 text conditioner, t2va (text-only) scope: the *unnormalized* hidden_states[50]
# of the 64-layer Qwen3-VL decoder (no LM head, no final norm), vs the HF reference on the
# released weights. Large-host test: ~62 GiB of shards and RAM; skips when unavailable.

import pytest
import torch
import transformers
from loguru import logger

import ttnn

from ....encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder, create_rope_tensors
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor
from .common import CONDITIONER_SUBFOLDER, conditioner_checkpoint_dir, load_reference_conditioner

# scoped: only these text_encoder files must be present for the test to run
_PATTERNS = [
    f"{CONDITIONER_SUBFOLDER}/config.json",
    f"{CONDITIONER_SUBFOLDER}/*.safetensors",
    f"{CONDITIONER_SUBFOLDER}/model.safetensors.index.json",
]


def _rope_params(text_config):
    """transformers >=5 has no top-level rope_theta, and a `dict.get` default evaluates eagerly."""
    params = getattr(text_config, "rope_parameters", None) or text_config.rope_scaling
    theta = params["rope_theta"] if "rope_theta" in params else text_config.rope_theta
    return theta, params["mrope_section"], params.get("mrope_interleaved", False)


def _reference_lm(path: str):
    hf = load_reference_conditioner(path)
    lm = hf.language_model if hasattr(hf, "language_model") else hf.model.language_model
    return lm.eval()


# axis-0 CCL is covered by test_qwen3vl_decoder_block.py; no FSDP: a Blackhole chip fits TP=4 without it
@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis", "num_links"),
    [
        pytest.param((4, 8), (4, 8), 1, 2, id="tp8_axis1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}], indirect=True
)
@pytest.mark.parametrize("seq_len", [128])
def test_minimax_h3_text_conditioner(
    *, mesh_device: ttnn.MeshDevice, submesh_shape, tp_axis, num_links: int, seq_len: int
) -> None:
    """The layer-50 tap MiniMax-H3 conditions on, under TP, against the HF golden."""
    from diffusers.modular_pipelines.minimax_h3.packing import MINIMAX_H3_TEXT_ENCODER_LAYER as TAP

    torch.manual_seed(0)
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    tp_factor = tuple(submesh.shape)[tp_axis]

    path = conditioner_checkpoint_dir(_PATTERNS)
    text_config = transformers.AutoConfig.from_pretrained(path).text_config
    assert text_config.num_hidden_layers > TAP, (
        f"MiniMax-H3 conditions on hidden_states[{TAP}], which needs more than {TAP} decoder layers, "
        f"but the conditioner config declares {text_config.num_hidden_layers}."
    )

    lm = _reference_lm(path)
    cfg = lm.config
    # head_dim is explicit (128), not hidden_size // heads (80); deriving it builds the wrong rotary width
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    rope_theta, mrope_section, mrope_interleaved = _rope_params(cfg)
    assert (
        sum(mrope_section) == head_dim // 2
    ), f"mrope_section {mrope_section} sums to {sum(mrope_section)}, expected head_dim//2 = {head_dim // 2}"

    ids = torch.randint(0, cfg.vocab_size, (1, seq_len))

    # chunked rope == interleaved only while all three MRoPE axes agree (text-only); assert, don't assume
    if mrope_interleaved:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

        text_positions = torch.arange(seq_len).view(1, 1, -1).expand(3, 1, -1)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        freqs = (inv_freq[None, None, :, None].expand(3, 1, -1, 1) @ text_positions[:, :, None, :].float()).transpose(
            2, 3
        )
        interleaved = Qwen3VLTextRotaryEmbedding(cfg).apply_interleaved_mrope(freqs.clone(), mrope_section)
        assert torch.equal(interleaved, freqs[0]), (
            "interleaved MRoPE is not a no-op for these position ids, so the chunked layout in "
            "`create_rope_tensors` no longer reproduces the reference. A vision-bearing request has "
            "reached a text-only code path."
        )

    # golden: a hook, not output_hidden_states, so the capture is unambiguously the layer output
    captured: dict[int, torch.Tensor] = {}
    handle = lm.layers[TAP].register_forward_hook(
        lambda m, i_, o: captured.__setitem__(TAP, (o[0] if isinstance(o, tuple) else o).detach())
    )
    with torch.no_grad():
        out = lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    handle.remove()
    golden = captured[TAP].float()
    assert golden.shape == (1, seq_len, cfg.hidden_size)

    enc = Qwen3VlTextEncoder(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        hidden_act=cfg.hidden_act,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=rope_theta,
        mrope_section=mrope_section,
        head_dim=head_dim,
        # activation_layers=None would return the *normalized* final state, which H3 does not use
        activation_layers=(TAP,),
        device=submesh,
        parallel_config=EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis)),
        ccl_manager=CCLManager(submesh, num_links=num_links, topology=ttnn.Topology.Linear),
    )
    enc.load_torch_state_dict(lm.state_dict())

    cos, sin = create_rope_tensors(1, seq_len, None, head_dim, rope_theta, mrope_section)
    tt_ids = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=submesh)
    tt_caps = enc.forward(
        tt_ids,
        attention_mask=None,
        pos_embeds=(bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh)),
    )

    assert len(tt_caps) == 1, f"expected a single tap at layer {TAP}, got {len(tt_caps)}"
    actual = tensor.to_torch(tt_caps[0], mesh_axes=[None, None, None])

    logger.info(f"minimax-h3 conditioner TP={tp_factor} axis={tp_axis} layer {TAP} of {cfg.num_hidden_layers}:")
    assert actual.shape[-2:] == (seq_len, cfg.hidden_size)
    assert_quality(golden, actual, pcc=0.99)

    assert not torch.allclose(golden, out.last_hidden_state.float(), atol=1e-2), (
        f"layer {TAP} output is indistinguishable from the normalized final hidden state; the tap is "
        "not exercising the mid-stack read MiniMax-H3 depends on."
    )
