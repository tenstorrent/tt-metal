# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace reference for the DFlash drafter, ``meta-models/Muse-Glimmer-30B-assistant``.

Same contract as :mod:`reference` for the target model: the real ``transformers``
classes are instantiated directly so the TTNN implementation is compared against
actual HF math rather than a re-implementation of it.

Two facts about this checkpoint are load-bearing and are asserted rather than
assumed, because both are easy to get silently wrong:

1. **The drafter has no ``embed_tokens`` and no ``lm_head``.**  Its 58 tensors are
   5 decoder layers + ``encoder.fc`` + ``encoder.output_norm_enc`` + ``norm``.
   Input embeddings come from the *target's* table via a plain ``F.embedding``
   lookup (deliberately *not* the target's normalised embedding path), and
   candidate logits come from the *target's* ``lm_head``.  A drafter that
   allocates its own would be 1.3 B parameters heavier and produce garbage.

2. **Attention over the diffusion window is bidirectional, not causal.**
   ``MuseGlimmerAssistantAttention.is_causal`` is ``False`` and the model builds
   ``create_bidirectional_sliding_window_mask``.  The 16 block positions attend
   to each other in both directions and causally to the context.  Porting this
   with the target's causal mask silently degrades acceptance rate rather than
   failing, which is the worst possible failure mode.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.muse_glimmer_assistant.modeling_muse_glimmer_assistant import MuseGlimmerAssistantModel

DRAFT_MODEL_ID = "meta-models/Muse-Glimmer-30B-assistant"
TARGET_MODEL_ID = "meta-models/Muse-Glimmer-30B"

#: The drafter's full tensor contract, verified against the safetensors header.
DRAFT_LAYER_WEIGHT_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
DRAFT_GLOBAL_WEIGHTS = (
    "encoder.fc.weight",
    "encoder.output_norm_enc.weight",
    "norm.weight",
)

#: Pinned architecture.  A checkpoint revision that moves any of these changes the
#: port's correctness, so it must fail loudly here rather than drift.
EXPECTED_CONFIG = {
    "block_size": 16,
    "mask_token_id": 201818,
    "target_layer_ids": [1, 13, 25, 37, 49],
    "num_hidden_layers": 5,
    "hidden_size": 6656,
    "intermediate_size": 19968,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "sliding_window": 2048,
    "rms_norm_eps": 1e-05,
    "max_position_embeddings": 131072,
}


@lru_cache(maxsize=1)
def draft_config():
    config = AutoConfig.from_pretrained(DRAFT_MODEL_ID, local_files_only=True)
    config._attn_implementation = "sdpa"
    _assert_pinned(config)
    return config


def _assert_pinned(config) -> None:
    for key, expected in EXPECTED_CONFIG.items():
        actual = getattr(config, key)
        if isinstance(expected, list):
            actual = list(actual)
        if actual != expected:
            raise AssertionError(f"drafter config drifted: {key} is {actual!r}, port assumes {expected!r}")
    if list(config.layer_types) != ["sliding_attention"] * config.num_hidden_layers:
        raise AssertionError(
            f"drafter layer_types drifted: {list(config.layer_types)!r}; "
            "the port assumes every drafter layer is sliding_attention"
        )
    # The rotary base lives on `rope_parameters`, exactly as it does on the target.
    # See the target port's work_log: `layer_rope_theta` there is a NoPE *gate*, not a base.
    if config.rope_parameters["rope_theta"] != 500000.0:
        raise AssertionError(f"drafter rope_theta drifted: {config.rope_parameters['rope_theta']}")


def draft_snapshot_dir() -> Path:
    """Cache snapshot holding the drafter weights.

    Resolved by looking for the weight file rather than trusting ``refs/main``,
    for the same reason the target's loader does: the default revision of a repo
    can be metadata-only.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    repo = Path(HF_HUB_CACHE) / f"models--{DRAFT_MODEL_ID.replace('/', '--')}"
    candidates = sorted(repo.glob("snapshots/*/model.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"no cached drafter weights for {DRAFT_MODEL_ID} under {repo}")
    return candidates[0].parent


@lru_cache(maxsize=1)
def draft_state_dict() -> dict[str, torch.Tensor]:
    """Every drafter tensor, and a check that the set is exactly what we expect."""
    from safetensors import safe_open

    path = draft_snapshot_dir() / "model.safetensors"
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as handle:
        for key in handle.keys():
            state_dict[key] = handle.get_tensor(key)

    config = draft_config()
    expected = set(DRAFT_GLOBAL_WEIGHTS)
    for layer_idx in range(config.num_hidden_layers):
        for suffix in DRAFT_LAYER_WEIGHT_SUFFIXES:
            expected.add(f"layers.{layer_idx}.{suffix}")
    actual = set(state_dict)
    if actual != expected:
        raise AssertionError(
            "drafter tensor set drifted.\n"
            f"  missing: {sorted(expected - actual)}\n"
            f"  unexpected: {sorted(actual - expected)}"
        )
    # Fact (1) from the module docstring, enforced.
    for banned in ("embed_tokens", "lm_head"):
        if any(banned in key for key in actual):
            raise AssertionError(f"drafter unexpectedly ships {banned}; the port assumes it reuses the target's")
    return state_dict


def reference_model(dtype: torch.dtype = torch.bfloat16) -> MuseGlimmerAssistantModel:
    """The real HF drafter, on real weights, ready for a forward.

    Must go through ``from_pretrained``.  Building on ``meta`` and then
    ``to_empty()`` + ``load_state_dict(assign=True)`` looks equivalent and is
    not: ``MuseGlimmerAssistantRotaryEmbedding.inv_freq`` is a **non-persistent**
    buffer, so it is absent from the state dict, ``to_empty`` gives it
    uninitialised memory, and nothing ever fills it.  The model then runs
    happily with a garbage RoPE table and produces plausible-looking but wrong
    activations - goldens generated that way silently grade the port against
    noise.  :func:`_assert_rope_initialised` makes that failure loud.
    """
    model = MuseGlimmerAssistantModel.from_pretrained(
        str(draft_snapshot_dir()), dtype=dtype, local_files_only=True, device_map="cpu"
    ).eval()
    _assert_rope_initialised(model)
    return model


def _assert_rope_initialised(model: MuseGlimmerAssistantModel) -> None:
    """``inv_freq[0]`` is ``base ** 0 == 1.0`` for every well-formed default RoPE."""
    inv_freq = model.rotary_emb.inv_freq.float()
    expected = 1.0 / (
        draft_config().rope_parameters["rope_theta"]
        ** (torch.arange(0, model.config.head_dim, 2, dtype=torch.float32) / model.config.head_dim)
    )
    if not torch.allclose(inv_freq, expected, rtol=1e-4, atol=1e-6):
        raise AssertionError(
            "drafter rotary inv_freq is not the default RoPE table - it was probably never "
            f"initialised (non-persistent buffer).\n  got[:4]      {inv_freq[:4].tolist()}\n"
            f"  expected[:4] {expected[:4].tolist()}"
        )


def synthetic_inputs(
    *,
    context_len: int,
    seed: int = 20260816,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Deterministic drafter inputs for a given accepted-context length.

    ``context_hidden_states`` is the concatenation over ``target_layer_ids`` of the
    target's per-layer outputs, so its last dim is ``5 * hidden_size == 33280``.
    """
    config = draft_config()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fan_in = len(config.target_layer_ids) * config.hidden_size
    return {
        "noise_embeds": torch.normal(
            0.0, 0.02, (1, config.block_size, config.hidden_size), generator=generator, dtype=torch.float32
        ).to(dtype),
        "context_hidden_states": torch.normal(
            0.0, 1.0, (1, context_len, fan_in), generator=generator, dtype=torch.float32
        ).to(dtype),
    }


def reference_forward(model, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Run the drafter and capture every tensor the TTNN port must match.

    **This must mirror how ``DFlashTokenCandidateGenerator`` actually calls the
    model, not merely call it in a way that runs.**  The driver passes an explicit
    ``attention_mask`` and a ``DFlashCache``; do neither and
    ``create_bidirectional_sliding_window_mask`` takes its
    ``allow_is_bidirectional_skip`` path and returns ``None`` — the model then
    runs with **no sliding window at all**.  That is invisible for any context
    shorter than the 2048 window and silently wrong beyond it: goldens generated
    the lazy way scored a correct port at PCC 0.93 for context 4096, while an
    unwindowed reimplementation scored 0.99997.

    The cache is what makes the mask the right *size*.  The mask is built inside
    ``forward`` before K/V are appended, so with an empty cache the base
    ``kv_length`` is just ``block_size``; ``DFlashCache.get_mask_sizes`` adds
    ``_previous_number_of_accepted_tokens`` to span the context positions that
    the attention concatenates in afterwards.
    """
    from transformers.cache_utils import DFlashCache

    context_len = inputs["context_hidden_states"].shape[1]
    block = inputs["noise_embeds"].shape[1]
    cache = DFlashCache(config=model.config)
    cache.set_previous_accepted_tokens(context_len)
    attention_mask = torch.ones(1, context_len + block, dtype=torch.long)

    captured: dict[str, torch.Tensor] = {}

    handles = [
        model.encoder.register_forward_hook(
            lambda _m, _i, out: captured.__setitem__("encoder_out", out.detach().clone())
        )
    ]
    for idx, layer in enumerate(model.layers):
        handles.append(
            layer.register_forward_hook(
                lambda _m, _i, out, idx=idx: captured.__setitem__(f"layer{idx}_out", out.detach().clone())
            )
        )

    try:
        with torch.no_grad():
            out = model(
                noise_embeds=inputs["noise_embeds"],
                context_hidden_states=inputs["context_hidden_states"],
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    captured["last_hidden_state"] = out.last_hidden_state.detach().clone()
    return captured


def golden_path() -> Path:
    return Path(__file__).with_name("dflash_goldens.pt")


def config_fingerprint() -> dict:
    """Recorded next to the goldens so a stale golden set is detectable."""
    config = draft_config()
    return {
        "draft_model_id": DRAFT_MODEL_ID,
        "snapshot": draft_snapshot_dir().name,
        **{key: getattr(config, key) for key in EXPECTED_CONFIG},
        "rope_theta": config.rope_parameters["rope_theta"],
    }


def _main() -> None:
    torch.manual_seed(0)
    model = reference_model()
    goldens: dict[str, dict] = {"_fingerprint": config_fingerprint()}
    for context_len in (1, 16, 128, 2048, 4096):
        inputs = synthetic_inputs(context_len=context_len)
        captured = reference_forward(model, inputs)
        # Inputs are NOT stored: they are a pure function of (context_len, seed) via
        # `synthetic_inputs`, and the ctx4096 context alone is 272 MB.  Consumers
        # rebuild them; the fingerprint guards against a drifting definition.
        goldens[f"ctx{context_len}"] = {"outputs": captured}
        last = captured["last_hidden_state"]
        print(
            f"ctx={context_len:5d}  last_hidden_state {tuple(last.shape)} "
            f"mean={last.float().mean():+.6f} std={last.float().std():.6f}"
        )
    torch.save(goldens, golden_path())
    print(f"\nwrote {golden_path()}")
    print(json.dumps(goldens["_fingerprint"], indent=2))


if __name__ == "__main__":
    _main()
