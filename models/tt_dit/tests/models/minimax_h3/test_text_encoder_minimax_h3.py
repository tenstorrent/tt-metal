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
from huggingface_hub import snapshot_download
from loguru import logger

import ttnn

from ....encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder, create_rope_tensors
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

_LOCAL_MIRROR = "/data/cglagovich/MiniMax-H3-diffusers"
_HF_REPO = "MiniMaxAI/MiniMax-H3"
_SUBFOLDER = "text_encoder"
# Scoped to the conditioner: the repository carries ~190 GB across three partitions
# (`transformer/`, `transformer_ref/`, `text_encoder/`) and an unscoped `snapshot_download`
# would pull all of it.
_PATTERNS = [
    f"{_SUBFOLDER}/config.json",
    f"{_SUBFOLDER}/*.safetensors",
    f"{_SUBFOLDER}/model.safetensors.index.json",
]


def _conditioner_dir() -> str:
    """The directory holding the Qwen3-VL conditioner.

    `MINIMAX_H3_REPO` (a local directory or a Hub repo id), then the local mirror, then a scoped Hub
    snapshot. A missing checkpoint is a skip, not a failure: there is nothing to compare against, and
    that is an environment gap rather than a defect in the port.
    """
    try:
        ref = os.environ.get("MINIMAX_H3_REPO", "").strip()
        if ref and os.path.isdir(ref):
            root = ref
        elif not ref and os.path.isdir(_LOCAL_MIRROR):
            root = _LOCAL_MIRROR
        else:
            repo_id = ref or _HF_REPO
            logger.info(f"MiniMax-H3 conditioner not local; fetching {_PATTERNS} from {repo_id}")
            root = snapshot_download(repo_id=repo_id, allow_patterns=_PATTERNS)
        return os.path.join(root, _SUBFOLDER)
    except Exception as exc:  # noqa: BLE001 - transport/auth/gating failures are a skip, not a failure
        pytest.skip(
            f"MiniMax-H3 conditioner unavailable (tried $MINIMAX_H3_REPO, {_LOCAL_MIRROR}, then " f"{_HF_REPO}): {exc}"
        )


def _rope_params(text_config):
    """`(rope_theta, mrope_section, mrope_interleaved)`.

    transformers >=5 moved `rope_theta` *into* `rope_parameters` and dropped the top-level attribute,
    so it must not be referenced as a `dict.get` default -- Python evaluates defaults eagerly and
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
    hf, info = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, output_loading_info=True
    )
    # Prove the shipped weights actually landed, rather than leaving parts of the reference on its
    # fresh init -- a silently partial load is the one way this comparison could go green without
    # having tested the checkpoint.
    # `loading_info` values are *sets*, so they are sorted before slicing; indexing a set raises, and
    # this runs on the failure path where a crash would hide the mismatch it is meant to report.
    bad = {k: sorted(info[k])[:5] for k in ("missing_keys", "unexpected_keys", "mismatched_keys") if info[k]}
    assert not bad, f"conditioner load key mismatch: {bad}"
    lm = hf.language_model if hasattr(hf, "language_model") else hf.model.language_model
    return lm.eval()


# Galaxy (32 chips). Both configs put TP on the size-8 axis, so the ~32B conditioner shards to
# ~7.5 GiB/device; TP on axis 0 and axis 1 take different CCL paths, which is what the two cover.
#
# No FSDP: `is_fsdp` stays at its False default, so weights are replicated on the non-TP axis rather
# than sharded across it. FSDP was required on a Wormhole 2x4, where TP=4 puts 14.9 GiB of weights on
# a 12 GiB chip; a Blackhole chip is 31.9 GiB, so even TP=4 fits and FSDP buys nothing here. The cost
# of skipping it is a 4x weight replication across the non-TP axis -- load-time bandwidth, not
# capacity.
#
# num_links=2 is the Blackhole Galaxy convention (Wormhole Galaxy uses 4); a bandwidth knob, not a
# correctness one.
@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis", "num_links"),
    [
        pytest.param((4, 8), (4, 8), 1, 2, id="tp8_axis1"),
        pytest.param((8, 4), (8, 4), 0, 2, id="tp8_axis0"),
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

    path = _conditioner_dir()
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
    # vision tokens). When the vision tower lands the axes diverge and this assertion will fail --
    # which is the intended signal that the chunked path is no longer valid.
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
    # The measured margin on the released weights is wide: PCC 99.9993%, RMSE/sigma 0.4%.
    assert_quality(golden, actual, pcc=0.99)

    # The tap must be the mid-stack layer output, not the normalized final state. Without this, an
    # encoder that quietly fell back to `activation_layers=None` would still pass the PCC check
    # whenever the two happened to correlate.
    assert not torch.allclose(golden, out.last_hidden_state.float(), atol=1e-2), (
        f"layer {TAP} output is indistinguishable from the normalized final hidden state; the tap is "
        "not exercising the mid-stack read MiniMax-H3 depends on."
    )
