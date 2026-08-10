# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared fp32 CPU reference builders for the ACE-Step 1.5 DiT PCC tests (Block 1).

Two sources of truth are supported and the tests accept either:

*   **random-init reference** (default) — build the `diffusers` module, replace every
    parameter with seeded pseudo-random values, run it on CPU in fp32, and compare. This is
    the ``tt_dit`` house convention for block-level tests ("Block-level tests use random-init
    reference weights; real-weight tests gate on an env var") and makes every test in this
    directory self-contained: no dependency on Block 0's golden dump.
*   **golden replay** — set ``ACE_STEP_DIT_GOLDEN=1`` and the tests load tensors from
    ``golden/dit/`` instead. See :data:`GOLDEN_DIR` and :func:`load_golden` for the expected
    naming; every golden is fp32, ``torch.save``d, seed 1234 (master doc §5b).

Nothing in this file touches a device.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from diffusers.models.transformers.ace_step_transformer import (
    AceStepAttention,
    AceStepMLP,
    AceStepTimestepEmbedding,
    AceStepTransformer1DModel,
    AceStepTransformerBlock,
    _create_4d_mask,
)

from models.experimental.ace_step_v15.tt.ttnn_ace_step_common import AceStepDiTConfig

#: Block 0 owns the dump: ``golden/dit/s<S>/transformer.<module path>.{in0,out,out0,out1,
#: kw_<argname>}.pt`` plus ``meta.pt``. All fp32, seed 1234, real converted weights.
GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden" / "dit"

#: Diffusers-format ACE-Step 1.5 checkpoint, same default as ``reference/ace_step_ref.py``.
PIPELINE_PATH = os.getenv("ACE_STEP_PIPELINE", "/localdev/acicovic/ace_step_diffusers")

#: Reference durations (master doc §5b): duration = 2.56 * k gives S = 32 * k.
SEQ_LEN_UNIT = 32  # 2.56 s  — op-level
SEQ_LEN_BLOCK = 128  # 10.24 s — single block
SEQ_LEN_BANDED = 256  # 20.48 s — the sliding window is a no-op below S ~= 130
SEQ_LEN_E2E = 768  # 61.44 s — end-to-end


def use_golden() -> bool:
    return os.environ.get("ACE_STEP_DIT_GOLDEN", "0") not in ("0", "", "false", "False")


class DitGoldens:
    """Read-only view of ``golden/dit/s<S>/`` as dumped by ``reference/dump_goldens.py``.

    Keys are given **without** the ``transformer.`` prefix, e.g. ``"layers.0.out"`` or
    ``"kw_hidden_states"``. Every tensor is returned fp32.

    Naming notes worth knowing when mapping to the TTNN capture keys:

    * ``layers.{i}.self_attn.kw_hidden_states`` is the **adaLN-modulated** norm output — i.e.
      the TTNN ``self_attn_norm_modulated`` capture, not ``self_attn_norm.out`` (which is the
      bare RMSNorm before modulation). Same for ``cross_attn.kw_hidden_states`` and
      ``mlp.in0``.
    * ``layers.{i}.kw_temb`` is ``timestep_proj_t + timestep_proj_r``, so the per-step chunks
      the TTNN model consumes are ``kw_temb - time_embed_r.out1``.
    * ``layers.{i}.kw_attention_mask`` exists only for even (``sliding_attention``) layers;
      odd layers genuinely get ``None``.
    * ``proj_in_conv`` / ``proj_out_conv`` tensors are **NCL** ``[B, C, T]``.
    * ``norm_q``/``norm_k`` in/out are ``[B, S, heads, head_dim]``; TTNN uses
      ``[B, heads, S, head_dim]``.
    """

    def __init__(self, seq_len: int) -> None:
        self.seq_len = seq_len
        self.dir = GOLDEN_DIR / f"s{seq_len}"
        if not (self.dir / "meta.pt").exists():
            msg = (
                f"golden directory {self.dir} is missing or has no meta.pt. Block 0 "
                f"(`reference/dump_goldens.py`) owns the dump; run it, or unset "
                f"ACE_STEP_DIT_GOLDEN to use the random-init reference."
            )
            raise FileNotFoundError(msg)
        self.meta = torch.load(self.dir / "meta.pt", map_location="cpu", weights_only=False)

    def _path(self, name: str) -> Path:
        return self.dir / f"transformer.{name}.pt"

    def has(self, name: str) -> bool:
        return self._path(name).exists()

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __getitem__(self, name: str) -> torch.Tensor:
        path = self._path(name)
        if not path.exists():
            msg = f"golden tensor {path} not found"
            raise KeyError(msg)
        return torch.load(path, map_location="cpu", weights_only=False).to(torch.float32)

    def get(self, name: str, default=None):
        return self[name] if self.has(name) else default


def real_dit_state_dict(prefix: str | None = None, *, path: str | None = None) -> dict[str, torch.Tensor]:
    """Load the real converted DiT weights as fp32, optionally only one subtree.

    Reads directly from the sharded safetensors via the index, so ``prefix="layers.3"``
    costs a few MB rather than the 6.3 GB of the whole 1575 M-parameter model. Returned keys
    have ``prefix.`` stripped, ready to hand to ``load_torch_state_dict``.
    """
    import json

    from safetensors import safe_open

    root = Path(path or PIPELINE_PATH) / "transformer"
    index_path = root / "diffusion_pytorch_model.safetensors.index.json"
    if not index_path.exists():
        msg = (
            f"{index_path} not found. Set $ACE_STEP_PIPELINE to a diffusers-format ACE-Step "
            f"1.5 directory (Block 0 produces it), or unset ACE_STEP_DIT_GOLDEN."
        )
        raise FileNotFoundError(msg)
    weight_map = json.loads(index_path.read_text())["weight_map"]

    wanted = {k: shard for k, shard in weight_map.items() if prefix is None or k.startswith(f"{prefix}.")}
    by_shard: dict[str, list[str]] = {}
    for key, shard in wanted.items():
        by_shard.setdefault(shard, []).append(key)

    out: dict[str, torch.Tensor] = {}
    strip = 0 if prefix is None else len(prefix) + 1
    for shard, keys in by_shard.items():
        with safe_open(str(root / shard), framework="pt", device="cpu") as handle:
            for key in keys:
                out[key[strip:]] = handle.get_tensor(key).to(torch.float32)
    if not out:
        msg = f"no weights matched prefix {prefix!r} in {root}"
        raise KeyError(msg)
    return out


# --------------------------------------------------------------------------------------- #
#                                   random initialisation                                  #
# --------------------------------------------------------------------------------------- #


def randomize_(module: torch.nn.Module, *, seed: int = 1234) -> torch.nn.Module:
    """Fill every parameter with seeded pseudo-random values, in fp32.

    Deliberately does **not** leave RMSNorm weights at their ``ones`` initialisation: an
    all-ones affine weight would make the norm tests pass even if the weight were dropped on
    the floor. Scaling follows fan-in so activations stay O(1) through 24 layers, which keeps
    PCC meaningful (a diverging reference makes every downstream PCC look great).
    """
    generator = torch.Generator().manual_seed(seed)

    def rand(shape: torch.Size, scale: float) -> torch.Tensor:
        return torch.randn(tuple(shape), generator=generator, dtype=torch.float32) * scale

    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("scale_shift_table"):
                # Reference init: randn(1, n, hidden) / hidden ** 0.5
                param.data = rand(param.shape, param.shape[-1] ** -0.5)
            elif param.dim() == 1 and not name.endswith("bias"):
                # RMSNorm affine weight (hidden_size or head_dim wide).
                param.data = 1.0 + rand(param.shape, 0.05)
            elif name.endswith("bias"):
                param.data = rand(param.shape, 0.02)
            else:
                fan_in = max(1, param.numel() // param.shape[0])
                param.data = rand(param.shape, fan_in**-0.5)
    return module.to(torch.float32).eval()


# --------------------------------------------------------------------------------------- #
#                                   reference modules                                      #
# --------------------------------------------------------------------------------------- #


def reference_kwargs(config: AceStepDiTConfig, *, num_hidden_layers: int | None = None) -> dict:
    return {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": num_hidden_layers or config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "in_channels": config.in_channels,
        "audio_acoustic_hidden_dim": config.audio_acoustic_hidden_dim,
        "patch_size": config.patch_size,
        "rope_theta": config.rope_theta,
        "attention_bias": config.attention_bias,
        "rms_norm_eps": config.rms_norm_eps,
        "sliding_window": config.sliding_window,
    }


def reference_attention(config: AceStepDiTConfig, *, is_cross: bool, seed: int = 1234) -> AceStepAttention:
    module = AceStepAttention(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        bias=config.attention_bias,
        eps=config.rms_norm_eps,
        sliding_window=None if is_cross else config.sliding_window,
        is_cross_attention=is_cross,
    )
    return randomize_(module, seed=seed)


def reference_mlp(config: AceStepDiTConfig, *, seed: int = 1234) -> AceStepMLP:
    return randomize_(AceStepMLP(config.hidden_size, config.intermediate_size), seed=seed)


def reference_timestep_embedding(config: AceStepDiTConfig, *, seed: int = 1234) -> AceStepTimestepEmbedding:
    module = AceStepTimestepEmbedding(
        in_channels=config.time_embed_in_channels,
        time_embed_dim=config.hidden_size,
        scale=config.time_embed_scale,
    )
    return randomize_(module, seed=seed)


def reference_block(config: AceStepDiTConfig, *, sliding: bool, seed: int = 1234) -> AceStepTransformerBlock:
    module = AceStepTransformerBlock(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        intermediate_size=config.intermediate_size,
        attention_bias=config.attention_bias,
        rms_norm_eps=config.rms_norm_eps,
        sliding_window=config.sliding_window if sliding else None,
        use_cross_attention=True,
    )
    return randomize_(module, seed=seed)


def reference_model(
    config: AceStepDiTConfig, *, num_hidden_layers: int | None = None, seed: int = 1234
) -> AceStepTransformer1DModel:
    module = AceStepTransformer1DModel(**reference_kwargs(config, num_hidden_layers=num_hidden_layers))
    return randomize_(module, seed=seed)


def sliding_mask(seq_len: int, sliding_window: int) -> torch.Tensor:
    """The reference's dense ``[1, 1, S, S]`` additive band, ``|i - j| <= sliding_window``.

    Built with the reference's own ``_create_4d_mask``, so there is no chance of the test
    encoding a different window convention than the model it is checking.
    """
    return _create_4d_mask(
        seq_len=seq_len,
        dtype=torch.float32,
        device=torch.device("cpu"),
        sliding_window=sliding_window,
        is_sliding_window=True,
        is_causal=False,
    )


def rope_for(seq_len: int, config: AceStepDiTConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` as the reference builds them: ``[S, head_dim]`` fp32."""
    from diffusers.models.transformers.ace_step_transformer import _ace_step_rotary_freqs

    return _ace_step_rotary_freqs(seq_len, config.head_dim, config.rope_theta, torch.device("cpu"), torch.float32)


# --------------------------------------------------------------------------------------- #
#                              staged block oracle (per-stage PCC)                          #
# --------------------------------------------------------------------------------------- #


def block_stages(
    block: AceStepTransformerBlock,
    config: AceStepDiTConfig,
    *,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    encoder_hidden_states: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Re-run the reference block one documented step at a time, keyed like the TT capture.

    Uses the reference block's **own submodules** in the reference order, so this is an oracle
    for localisation rather than an independent re-derivation. ``test_dit_block_pcc`` asserts
    that ``block_stages(...)["out"]`` matches ``block(...)`` at PCC ~1.0 before trusting any of
    the intermediate keys.

    Head-split tensors are returned in the TTNN ``[B, heads, S, head_dim]`` layout, not the
    reference's ``[B, S, heads, head_dim]``.
    """
    import torch.nn.functional as F  # noqa: N812
    from diffusers.models.embeddings import apply_rotary_emb

    heads, kv_heads, head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    cos, sin = rope
    bhsd = lambda t: t.permute(0, 2, 1, 3)  # [B, S, H, D] -> [B, H, S, D]  # noqa: E731
    out: dict[str, torch.Tensor] = {}

    shift, scale, gate, c_shift, c_scale, c_gate = (block.scale_shift_table + temb).chunk(6, dim=1)

    # --- steps 1-8: self-attention ------------------------------------------------------ #
    h = block.self_attn_norm(hidden_states) * (1 + scale) + shift
    out["self_attn_norm_modulated"] = h
    attn = block.self_attn
    q = attn.to_q(h).unflatten(-1, (heads, head_dim))
    k = attn.to_k(h).unflatten(-1, (kv_heads, head_dim))
    v = attn.to_v(h).unflatten(-1, (kv_heads, head_dim))
    out["self_attn.q_pre_norm"] = bhsd(q)
    out["self_attn.k_pre_norm"] = bhsd(k)
    out["self_attn.v"] = bhsd(v)
    q = attn.norm_q(q)
    k = attn.norm_k(k)
    out["self_attn.q_normed"] = bhsd(q)
    out["self_attn.k_normed"] = bhsd(k)
    q = apply_rotary_emb(q, (cos, sin), use_real=True, use_real_unbind_dim=-2, sequence_dim=1)
    k = apply_rotary_emb(k, (cos, sin), use_real=True, use_real_unbind_dim=-2, sequence_dim=1)
    out["self_attn.q_rope"] = bhsd(q)
    out["self_attn.k_rope"] = bhsd(k)
    o = F.scaled_dot_product_attention(
        bhsd(q), bhsd(k), bhsd(v), attn_mask=attention_mask, scale=attn.scaling, enable_gqa=True
    )
    out["self_attn.sdpa"] = o
    o = attn.to_out[0](o.permute(0, 2, 1, 3).flatten(2, 3))
    out["self_attn.out"] = o
    x = hidden_states + o * gate  # BARE gate
    out["after_self_attn"] = x

    # --- steps 9-11: cross-attention ---------------------------------------------------- #
    h = block.cross_attn_norm(x)
    out["cross_attn_norm"] = h
    cross = block.cross_attn
    cq = cross.norm_q(cross.to_q(h).unflatten(-1, (heads, head_dim)))
    ck = cross.norm_k(cross.to_k(encoder_hidden_states).unflatten(-1, (kv_heads, head_dim)))
    cv = cross.to_v(encoder_hidden_states).unflatten(-1, (kv_heads, head_dim))
    out["cross_attn.q"] = bhsd(cq)
    out["cross_attn.k"] = bhsd(ck)
    out["cross_attn.v"] = bhsd(cv)
    co = F.scaled_dot_product_attention(
        bhsd(cq), bhsd(ck), bhsd(cv), attn_mask=None, scale=cross.scaling, enable_gqa=True
    )
    out["cross_attn.sdpa"] = co
    co = cross.to_out[0](co.permute(0, 2, 1, 3).flatten(2, 3))
    out["cross_attn.out"] = co
    x = x + co  # plain residual, no gate
    out["after_cross_attn"] = x

    # --- steps 12-14: SwiGLU MLP -------------------------------------------------------- #
    h = block.mlp_norm(x) * (1 + c_scale) + c_shift
    out["mlp_norm_modulated"] = h
    ff = block.mlp(h)
    out["mlp_out"] = ff
    out["out"] = x + ff * c_gate
    return out
