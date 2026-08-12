# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# MiniMax-H3 text conditioner: the Qwen3-VL read MiniMax-H3 performs, checked
# against the HF reference on the released weights.
#
# MiniMax-H3 conditions on ONE hidden state of its Qwen3-VL conditioner:
# `hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]`, i.e. the *unnormalized* output
# of decoder layer 50 of 64, mid-stack. The language-model head is never used and
# the final norm is never applied -- diffusers is explicit that the last hidden
# state of a stack truncated to 50 layers is post-norm and is NOT the
# conditioning MiniMax-H3 expects (`encoders.py::encode_prompt`).
#
# Scope: the `t2va` (text-only) task. For `t2va` no `pixel_values` reach the
# conditioner, so `Qwen3VLModel` never runs its vision tower and never injects
# deepstack features -- the conditioner degenerates to a plain text decoder,
# which is what `Qwen3VlTextEncoder` implements. `fl2va` / `ref2va` DO feed
# vision (embedding scatter + deepstack at decoder layers 0/1/2) and are out of
# scope until the vision tower is ported.
#
# The conditioner is ~32B params, bf16 both on disk and in the comparison, so
# this is a large-host test: it needs the ~62 GiB of shards resolvable and about
# that much RAM. It skips rather than fails when they are not.
# =============================================================================

import os

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

# Scoped to the conditioner: the repository carries ~190 GB across three partitions
# (`transformer/`, `transformer_ref/`, `text_encoder/`) and an unscoped `snapshot_download`
# would pull all of it.
_PATTERNS = [
    f"{CONDITIONER_SUBFOLDER}/config.json",
    f"{CONDITIONER_SUBFOLDER}/*.safetensors",
    f"{CONDITIONER_SUBFOLDER}/model.safetensors.index.json",
]

# `MINIMAX_H3_RUN_REF=0` skips the golden: no reference forward, no comparison, just our
# implementation, asserting only shapes and finiteness. Default is on, so a plain `pytest` run is a
# parity test.
#
# It is NOT a speed-up worth reaching for. The reference forward is memory-bandwidth bound at roughly
# 2s; what dominates these tests is `load_torch_state_dict` pushing ~32B of weights onto the mesh, which
# happens either way because our encoder takes its weights and config from the checkpoint.
#
# What it is for: exercising the device path when the golden is unavailable or untrusted (a different
# transformers version, say), keeping the CPU forward out of a `python -m tracy` capture, and
# smoke-testing plumbing changes. A green run under it proves nothing about accuracy.
RUN_REF = os.environ.get("MINIMAX_H3_RUN_REF", "1").strip().lower() not in {"0", "false", "no"}


def _rope_params(text_config):
    """`(rope_theta, mrope_section, mrope_interleaved)`.

    transformers >=5 keeps `rope_theta` inside `rope_parameters` with no top-level attribute, so it
    must not be referenced as a `dict.get` default -- Python evaluates defaults eagerly and
    `text_config.rope_theta` raises `AttributeError` on this config.
    """
    params = getattr(text_config, "rope_parameters", None) or text_config.rope_scaling
    theta = params["rope_theta"] if "rope_theta" in params else text_config.rope_theta
    return theta, params["mrope_section"], params.get("mrope_interleaved", False)


def _reference_lm(path: str):
    """The HF Qwen3-VL decoder stack MiniMax-H3 taps, on the released weights.

    Loaded through the full `Qwen3VLForConditionalGeneration` -- the class diffusers declares in its
    `ComponentSpec` -- and narrowed to `.model.language_model`. Keeping the whole model matches what
    `encoders.py` calls (`text_encoder.model(...)`, i.e. the `Qwen3VLModel` that owns both the vision
    tower and the decoder); with `pixel_values=None` that call touches the vision tower not at all,
    so the tap is the same tensor either way.

    The checkpoint is already bf16 on disk (1058/1058 tensors), so `dtype` loads it as-is rather than
    materializing fp32 first the way `from_config(...).to(bf16)` would.
    """
    hf = load_reference_conditioner(path)
    lm = hf.language_model if hasattr(hf, "language_model") else hf.model.language_model
    return lm.eval()


# Galaxy (32 chips). TP on the size-8 axis -- axis 1, the config the pipeline runs -- shards the
# ~32B conditioner to ~7.5 GiB/device. Only axis 1 runs here: the axis choice changes only which CCL
# path the collectives take, and that path is covered by
# `tests/encoders/qwen3vl/test_qwen3vl_decoder_block.py`'s tp8_axis0 case without a second ~32B
# weight load.
#
# No FSDP: `is_fsdp` stays at its False default, so weights are replicated on the non-TP axis rather
# than sharded across it. FSDP is needed on a Wormhole 2x4, where TP=4 puts 14.9 GiB of weights on
# a 12 GiB chip; a Blackhole chip has 31.9 GiB, so even TP=4 fits and FSDP buys nothing here. The cost
# of skipping it is a 4x weight replication across the non-TP axis -- load-time bandwidth, not
# capacity.
#
# num_links=2 is the Blackhole Galaxy convention (Wormhole Galaxy uses 4); a bandwidth knob, not a
# correctness one.
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
    # The tap index is a checkpoint contract, so it is imported rather than restated.
    from diffusers.modular_pipelines.minimax_h3.packing import MINIMAX_H3_TEXT_ENCODER_LAYER as TAP

    torch.manual_seed(0)
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    tp_factor = tuple(submesh.shape)[tp_axis]

    path = conditioner_checkpoint_dir(_PATTERNS)
    text_config = transformers.AutoConfig.from_pretrained(path).text_config
    # The guard `encoders.py::encode_prompt` raises: reading `hidden_states[50]` needs *more* than 50
    # decoder layers, because a stack truncated to exactly 50 ends post-norm.
    assert text_config.num_hidden_layers > TAP, (
        f"MiniMax-H3 conditions on hidden_states[{TAP}], which needs more than {TAP} decoder layers, "
        f"but the conditioner config declares {text_config.num_hidden_layers}."
    )

    lm = _reference_lm(path)
    cfg = lm.config
    # `head_dim` is explicit (128) and does NOT equal hidden_size // num_attention_heads (5120//64 =
    # 80) on this checkpoint, so it must be read, not derived. Deriving it silently builds the wrong
    # rotary width. This matches how HF itself resolves the dimension.
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    rope_theta, mrope_section, mrope_interleaved = _rope_params(cfg)
    assert (
        sum(mrope_section) == head_dim // 2
    ), f"mrope_section {mrope_section} sums to {sum(mrope_section)}, expected head_dim//2 = {head_dim // 2}"

    ids = torch.randint(0, cfg.vocab_size, (1, seq_len))

    # --- the rotary path is only exact for text-only conditioning; assert that, don't assume it ---
    # This checkpoint sets `mrope_interleaved=True`, and `create_rope_tensors` implements the
    # *chunked* ([TTT..HHH..WWW]) layout, not the interleaved ([THWTHW..]) one. The two coincide
    # exactly while all three MRoPE axes carry the same position, which is the case for `t2va` (no
    # vision tokens). A vision-bearing request makes the axes diverge and this assertion fail --
    # the intended signal that the chunked path stops reproducing the reference.
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

    # --- golden: the raw output of decoder layer TAP ---
    # A forward hook is used rather than `output_hidden_states=True` so what is captured is
    # unambiguously the layer output, with no final norm applied anywhere.
    golden, out = None, None
    if RUN_REF:
        captured: dict[int, torch.Tensor] = {}
        handle = lm.layers[TAP].register_forward_hook(
            lambda m, i_, o: captured.__setitem__(TAP, (o[0] if isinstance(o, tuple) else o).detach())
        )
        with torch.no_grad():
            out = lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        handle.remove()
        golden = captured[TAP].float()

        # MiniMax-H3 feeds this straight into `transformer.context_embedder`, whose `text_dim` is 5120.
        assert golden.shape == (1, seq_len, cfg.hidden_size)
    else:
        logger.warning("MINIMAX_H3_RUN_REF=0: golden skipped, running our implementation only")

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
        # Explicit: 5120 // 64 would give 80, not the checkpoint's 128.
        head_dim=head_dim,
        # A single tap. `activation_layers=None` would return the *normalized* final hidden state,
        # which is the conditioning MiniMax-H3 explicitly does not use.
        activation_layers=(TAP,),
        device=submesh,
        parallel_config=EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis)),
        ccl_manager=CCLManager(submesh, num_links=num_links, topology=ttnn.Topology.Linear),
        # `is_fsdp` left at its False default -- see the mesh-parametrize note above.
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
    if not RUN_REF:
        # No golden to compare against, so check only what can be checked without one.
        assert torch.isfinite(actual).all(), "our output contains NaN or Inf"
        logger.info(f"  no golden (MINIMAX_H3_RUN_REF=0); ours mean |x| {actual.abs().mean():.4f}")
        return

    # The measured margin on the released weights is wide: PCC 99.9993%, RMSE/sigma 0.4%.
    assert_quality(golden, actual, pcc=0.99)

    # The tap must be the mid-stack layer output, not the normalized final state. Without this, an
    # encoder that quietly fell back to `activation_layers=None` would still pass the PCC check
    # whenever the two happened to correlate.
    assert not torch.allclose(golden, out.last_hidden_state.float(), atol=1e-2), (
        f"layer {TAP} output is indistinguishable from the normalized final hidden state; the tap is "
        "not exercising the mid-stack read MiniMax-H3 depends on."
    )
