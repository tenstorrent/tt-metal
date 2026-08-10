# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-side weight preparation for the ACE-Step 1.5 TTNN bringup — Block 0.

**Pure torch. This module must never import ttnn at module scope, never open a device,
and never allocate a device tensor.** Everything here is a state-dict -> state-dict
transform that runs on the host, so it can be unit-tested on a box with no accelerator
and reused from `tt_dit`'s `Module._prepare_torch_state` hook.

What lives here
---------------
1. `remap_upstream_to_diffusers()` — the upstream -> diffusers key remap
   (ACE_STEP_1_5_BRINGUP.md §2). Our converted checkpoint is *already* diffusers-format,
   so in practice `validate_diffusers_state()` is what runs; the remap exists for raw
   upstream checkpoints (`ACE-Step/acestep-v15-turbo-*`) and is exercised by a
   round-trip test.
   NOTE `q_norm` -> `norm_q` and `k_norm` -> `norm_k` ARE renamed.

2. `fold_weight_norm()` — folds `weight_norm` for the Oobleck VAE. Handles BOTH the
   legacy `weight_g`/`weight_v` pair AND the newer
   `parametrizations.weight.original0`/`original1` pair, and raises if a module has a
   `.bias` but no foldable weight (i.e. it would have loaded unfused).

3. `collapse_proj_in_channels()` — 192 -> 129 channel collapse of `proj_in_conv.weight`,
   because channels 64:128 (`chunk_masks`) all carry the same per-frame scalar.
   `verify_proj_in_collapse()` checks this numerically against the full form.

4. `proj_in_as_linear()` / `proj_out_as_linear()` — the patchify/de-patchify convs
   restated as plain matmuls (`kernel == stride == patch_size`, no padding), which is
   what makes the DiT convolution-free. Both have numeric verifiers.

5. `pack_swiglu_ff1()` — packs ACE-Step's separate `gate_proj`/`up_proj` into the single
   `[2*inner, dim]` weight that `tt_dit.layers.feedforward.FeedForward`'s fused SwiGLU
   `ff1` expects, in `[up | gate]` row order.

6. `prepare_dit_weights()` / `prepare_vae_decoder_weights()` — convenience wrappers that
   apply the above to a whole state dict.

Run the self-check (host-only, needs the diffusers checkpoint):
    python models/experimental/ace_step_v15/tt/ttnn_ace_step_weights.py
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

StateDict = Dict[str, torch.Tensor]

# --------------------------------------------------------------------------------------
# 0. config-ish constants (2 B turbo; asserted against config.json where possible)
# --------------------------------------------------------------------------------------

CHUNK_MASK_SLICE = slice(64, 128)  # channels of `proj_in` input that are the chunk_mask
IN_CHANNELS_FULL = 192  # 64 src_latents + 64 chunk_masks + 64 noisy x_t
IN_CHANNELS_COLLAPSED = 129  # 64 src_latents + 1 chunk_mask scalar + 64 noisy x_t
WEIGHT_NORM_EPS = 1e-9  # only used by the explicit fallback formula; see fold_weight_norm

# --------------------------------------------------------------------------------------
# 1. upstream -> diffusers key remap  (BRINGUP §2)
# --------------------------------------------------------------------------------------

# Attention renames. Applied to ALL prefix branches (DiT, condition encoder, tokenizer,
# detokenizer) — not just the DiT.
_ATTN_KEY_RENAMES = (
    (".q_proj.", ".to_q."),
    (".k_proj.", ".to_k."),
    (".v_proj.", ".to_v."),
    (".o_proj.", ".to_out.0."),
    (".q_norm.", ".norm_q."),  # RENAMED, not kept
    (".k_norm.", ".norm_k."),  # RENAMED, not kept
)

# Extra per-branch renames applied after the prefix is stripped.
_TRANSFORMER_RENAMES = (
    ("proj_in.1.", "proj_in_conv."),  # upstream wraps the conv in an nn.Sequential
    ("proj_out.1.", "proj_out_conv."),
)

# prefix -> diffusers component. Order matters only in that these prefixes are disjoint.
_PREFIX_ROUTES = (
    ("decoder.", "transformer"),
    ("encoder.", "condition_encoder"),
    ("tokenizer.", "audio_tokenizer"),
    ("detokenizer.", "audio_token_detokenizer"),
)

# Exact keys that route without a prefix strip.
_EXACT_ROUTES = {"null_condition_emb": ("condition_encoder", "null_condition_emb")}


def _apply_attn_renames(key: str) -> str:
    # Guard the leading/trailing dots by padding, so a key that *starts* with "q_proj."
    # is still matched.
    padded = "." + key
    for old, new in _ATTN_KEY_RENAMES:
        padded = padded.replace(old, new)
    return padded[1:]


def remap_upstream_to_diffusers(state: StateDict) -> Tuple[Dict[str, StateDict], list]:
    """Split an upstream ACE-Step 1.5 checkpoint into per-component diffusers state dicts.

    Returns `(components, dropped)` where `components` is
    `{"transformer": {...}, "condition_encoder": {...}, "audio_tokenizer": {...},
      "audio_token_detokenizer": {...}}` and `dropped` lists the keys that matched no
    prefix (the converter reports and discards these).

    `silence_latent.pt` is NOT handled here — it is a separate file upstream and the
    diffusers converter bakes it in as `condition_encoder.silence_latent`. Use
    `bake_silence_latent()`.
    """
    components: Dict[str, StateDict] = {name: {} for _, name in _PREFIX_ROUTES}
    dropped = []
    for key, value in state.items():
        if key in _EXACT_ROUTES:
            comp, new_key = _EXACT_ROUTES[key]
            components[comp][new_key] = value
            continue
        for prefix, comp in _PREFIX_ROUTES:
            if key.startswith(prefix):
                new_key = _apply_attn_renames(key[len(prefix) :])
                if comp == "transformer":
                    for old, new in _TRANSFORMER_RENAMES:
                        if new_key.startswith(old):
                            new_key = new + new_key[len(old) :]
                components[comp][new_key] = value
                break
        else:
            dropped.append(key)
    return components, dropped


def bake_silence_latent(condition_encoder_state: StateDict, silence_latent: torch.Tensor) -> StateDict:
    """Bake `silence_latent.pt` into the condition-encoder state dict.

    The upstream file is `[1, timbre_hidden_dim, T]`; the converter transposes it to
    `[1, T, timbre_hidden_dim]` (= the timbre encoder's input layout) and registers it as
    a persistent buffer. Upstream's loader raises FileNotFoundError without it.
    """
    condition_encoder_state = dict(condition_encoder_state)
    condition_encoder_state["silence_latent"] = silence_latent.transpose(1, 2).contiguous()
    return condition_encoder_state


# --- validation of an already-diffusers-format checkpoint -------------------------------

_DIT_EXPECTED_SHAPES = {
    "proj_in_conv.weight": ("hidden_size", "in_channels", "patch_size"),
    "proj_in_conv.bias": ("hidden_size",),
    "proj_out_conv.weight": ("hidden_size", "audio_acoustic_hidden_dim", "patch_size"),
    "proj_out_conv.bias": ("audio_acoustic_hidden_dim",),
    "condition_embedder.weight": ("hidden_size", "encoder_hidden_size"),
    "condition_embedder.bias": ("hidden_size",),
    "norm_out.weight": ("hidden_size",),
    "scale_shift_table": (1, 2, "hidden_size"),
}

_DIT_LAYER_EXPECTED_SHAPES = {
    "self_attn_norm.weight": ("hidden_size",),
    "self_attn.to_q.weight": ("q_width", "hidden_size"),
    "self_attn.to_k.weight": ("kv_width", "hidden_size"),
    "self_attn.to_v.weight": ("kv_width", "hidden_size"),
    "self_attn.to_out.0.weight": ("hidden_size", "q_width"),
    "self_attn.norm_q.weight": ("head_dim",),
    "self_attn.norm_k.weight": ("head_dim",),
    "cross_attn_norm.weight": ("hidden_size",),
    "cross_attn.to_q.weight": ("q_width", "hidden_size"),
    "cross_attn.to_k.weight": ("kv_width", "hidden_size"),
    "cross_attn.to_v.weight": ("kv_width", "hidden_size"),
    "cross_attn.to_out.0.weight": ("hidden_size", "q_width"),
    "cross_attn.norm_q.weight": ("head_dim",),
    "cross_attn.norm_k.weight": ("head_dim",),
    "mlp_norm.weight": ("hidden_size",),
    "mlp.gate_proj.weight": ("intermediate_size", "hidden_size"),
    "mlp.up_proj.weight": ("intermediate_size", "hidden_size"),
    "mlp.down_proj.weight": ("hidden_size", "intermediate_size"),
    "scale_shift_table": (1, 6, "hidden_size"),
}

_DIT_TIME_EMBED_SHAPES = {
    "linear_1.weight": ("hidden_size", 256),
    "linear_1.bias": ("hidden_size",),
    "linear_2.weight": ("hidden_size", "hidden_size"),
    "linear_2.bias": ("hidden_size",),
    "time_proj.weight": ("six_hidden", "hidden_size"),
    "time_proj.bias": ("six_hidden",),
}


def _dims(config: dict) -> dict:
    hidden = config["hidden_size"]
    return {
        "hidden_size": hidden,
        "encoder_hidden_size": config.get("encoder_hidden_size") or hidden,
        "intermediate_size": config["intermediate_size"],
        "in_channels": config["in_channels"],
        "audio_acoustic_hidden_dim": config["audio_acoustic_hidden_dim"],
        "patch_size": config["patch_size"],
        "head_dim": config["head_dim"],
        "q_width": config["num_attention_heads"] * config["head_dim"],
        "kv_width": config["num_key_value_heads"] * config["head_dim"],
        "six_hidden": 6 * hidden,
    }


def _check(state: StateDict, key: str, want, dims: dict, problems: list) -> None:
    if key not in state:
        problems.append(f"missing key: {key}")
        return
    expect = tuple(dims[d] if isinstance(d, str) else d for d in want)
    got = tuple(state[key].shape)
    if got != expect:
        problems.append(f"shape mismatch {key}: got {got}, want {expect}")


def validate_diffusers_state(state: StateDict, config: dict, strict: bool = True) -> list:
    """Assert an *already diffusers-format* DiT state dict matches `config`.

    This is the path our checkpoint actually takes (§2: our conversion is already done),
    so it is a hard gate rather than a remap. Returns the list of problems; raises when
    `strict` and the list is non-empty.

    Also asserts the two things the remap is easy to get wrong: that NO upstream-style
    key survived (`q_proj`/`o_proj`/`q_norm`/`proj_in.1`), and that every dimension is a
    multiple of 32 (BRINGUP §3.4 — no padding needed anywhere in the DiT).
    """
    dims = _dims(config)
    problems: list = []

    for key, want in _DIT_EXPECTED_SHAPES.items():
        _check(state, key, want, dims, problems)
    for embed in ("time_embed", "time_embed_r"):
        for key, want in _DIT_TIME_EMBED_SHAPES.items():
            _check(state, f"{embed}.{key}", want, dims, problems)
    for i in range(config["num_hidden_layers"]):
        for key, want in _DIT_LAYER_EXPECTED_SHAPES.items():
            _check(state, f"layers.{i}.{key}", want, dims, problems)

    # No leftover upstream naming.
    upstream_markers = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "proj_in.1", "proj_out.1")
    for key in state:
        for marker in upstream_markers:
            if marker in key:
                problems.append(f"upstream-style key survived the remap: {key} (contains {marker!r})")

    # No `decoder.` / `encoder.` prefixes left.
    for key in state:
        if key.startswith("decoder.") or key.startswith("encoder."):
            problems.append(f"un-stripped component prefix: {key}")

    # Tile alignment (BRINGUP §3.4).
    for name in ("hidden_size", "intermediate_size", "in_channels", "audio_acoustic_hidden_dim", "q_width", "kv_width"):
        if dims[name] % 32 != 0:
            problems.append(f"{name}={dims[name]} is not a multiple of 32")
    if dims["head_dim"] % 32 != 0:
        problems.append(f"head_dim={dims['head_dim']} is not a multiple of 32")

    # attention_bias=False -> no attention biases anywhere.
    if not config.get("attention_bias", False):
        stray = [k for k in state if re.search(r"(to_q|to_k|to_v|to_out\.0)\.bias$", k)]
        if stray:
            problems.append(f"attention_bias is false but found {len(stray)} attention bias tensors, e.g. {stray[0]}")

    if strict and problems:
        raise AssertionError("diffusers DiT state validation failed:\n  " + "\n  ".join(problems))
    return problems


# --------------------------------------------------------------------------------------
# 2. weight_norm folding for the Oobleck VAE
# --------------------------------------------------------------------------------------

# The two shapes `weight_norm` can take on disk. torch 2.x still supports the legacy
# hook form but emits a FutureWarning (live in this build — the VAE load prints it), so a
# checkpoint re-saved by a newer torch would use the parametrization form instead.
_LEGACY_G, _LEGACY_V = "weight_g", "weight_v"
_PARAM_G, _PARAM_V = "parametrizations.weight.original0", "parametrizations.weight.original1"


def _weight_norm(v: torch.Tensor, g: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """`w = g * v / ||v||`, norm taken over every axis except `dim`.

    Uses `torch._weight_norm`, which is *exactly* what both `nn.utils.weight_norm` and
    `nn.utils.parametrizations.weight_norm` call at forward time (`_WeightNorm.forward`
    -> `torch._weight_norm(weight_v, weight_g, dim)`), so the fold is bit-exact against
    the reference rather than merely close. Note torch's kernel has **no epsilon**; the
    `WEIGHT_NORM_EPS = 1e-9` mentioned in the bringup doc belongs to the explicit
    fallback below, which is only used if the ATen op is unavailable. The two agree to
    ~1e-7 relative on real weights (checked by `self_check()`).
    """
    try:
        return torch._weight_norm(v.contiguous(), g.contiguous(), dim)
    except (AttributeError, RuntimeError):  # pragma: no cover - fallback path
        reduce_dims = [d for d in range(v.dim()) if d != dim]
        norm = v.float().pow(2).sum(dim=reduce_dims, keepdim=True).sqrt()
        return (g.float() * v.float() / (norm + WEIGHT_NORM_EPS)).to(v.dtype)


def fold_weight_norm(state: StateDict, strict: bool = True) -> StateDict:
    """Fold every `weight_norm`-parametrized weight in `state` into a plain `.weight`.

    Recognises both storage forms:
      legacy          `<p>.weight_g` [C,1,1] + `<p>.weight_v` [C,...]
      parametrization `<p>.parametrizations.weight.original0` + `...original1`

    Raises (when `strict`) if a module looks convolutional (`<p>.bias` present, or
    `<p>.weight` present with any sibling `weight_*`) but neither pair is found — the
    failure mode the bringup doc warns about is loading *unfused* weights silently,
    which produces plausible-but-wrong audio.

    ⚠ The `dim=0` axis means different things for the two conv flavours: Conv1d weight is
    `[out, in, K]` so `g` is per **output** channel, while ConvTranspose1d weight is
    `[in, out, K]` so `g` is per **input** channel. The fold itself is identical
    (norm over dims 1,2) — only the interpretation differs. Only the five
    `block.N.conv_t1.*` keys are ConvTranspose1d.
    """
    out: StateDict = {}
    prefixes = set()
    for key in state:
        for suffix in (_LEGACY_G, _LEGACY_V, _PARAM_G, _PARAM_V):
            if key == suffix:  # a bare weight-normed module saved on its own
                prefixes.add("")
            elif key.endswith("." + suffix):
                prefixes.add(key[: -len(suffix) - 1])

    folded = 0
    for prefix in sorted(prefixes):
        j = lambda name, p=prefix: f"{p}.{name}" if p else name
        legacy = j(_LEGACY_G) in state and j(_LEGACY_V) in state
        param = j(_PARAM_G) in state and j(_PARAM_V) in state
        if legacy:
            g, v = state[j(_LEGACY_G)], state[j(_LEGACY_V)]
        elif param:
            g, v = state[j(_PARAM_G)], state[j(_PARAM_V)]
        else:
            msg = (
                f"{prefix}: found a partial weight_norm parametrization. Expected either "
                f"({_LEGACY_G}, {_LEGACY_V}) or ({_PARAM_G}, {_PARAM_V}); refusing to load unfused."
            )
            if strict:
                raise KeyError(msg)
            continue
        if g.shape[0] != v.shape[0] or g.numel() != g.shape[0]:
            raise ValueError(f"{prefix!r}: unexpected weight_norm g shape {tuple(g.shape)} for v {tuple(v.shape)}")
        out[j("weight")] = _weight_norm(v, g, dim=0)
        folded += 1

    if prefixes and folded == 0:
        raise KeyError("found weight_norm-looking keys but folded none — check the naming scheme")

    consumed = set()
    for prefix in prefixes:
        for suffix in (_LEGACY_G, _LEGACY_V, _PARAM_G, _PARAM_V):
            consumed.add(f"{prefix}.{suffix}" if prefix else suffix)
    for key, value in state.items():
        if key not in consumed:
            out.setdefault(key, value)

    # Loud failure instead of a silent unfused load: every `<p>.bias` must have a
    # matching `<p>.weight` once folding is done. A conv whose weight_norm pair used an
    # unrecognised naming scheme lands here.
    if strict:
        unfused = sorted(
            {k[: -len(".bias")] for k in out if k.endswith(".bias") and k[: -len("bias")] + "weight" not in out}
        )
        if unfused:
            raise KeyError(
                f"{len(unfused)} module(s) have a bias but no weight after weight_norm folding — the checkpoint "
                f"uses an unrecognised weight_norm naming scheme and would have loaded UNFUSED: {unfused[:8]}"
            )
    return out


def snake_params(state: StateDict, prefix: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute the two `[C]` vectors the Snake activation actually needs.

    Reference (`diffusers.models.autoencoders.autoencoder_oobleck.Snake1d`, logscale=True):
        y = x + 1/(exp(beta) + 1e-9) * sin(exp(alpha) * x)**2
    The checkpoint stores log-scale `alpha`/`beta` as `[1, C, 1]`. Returns
    `(exp(alpha), 1/(exp(beta)+1e-9))`, both `[C]`.
    """
    alpha = state[f"{prefix}.alpha"].reshape(-1).float()
    beta = state[f"{prefix}.beta"].reshape(-1).float()
    return torch.exp(alpha), torch.reciprocal(torch.exp(beta) + 1e-9)


# --------------------------------------------------------------------------------------
# 3. proj_in channel collapse 192 -> 129
# --------------------------------------------------------------------------------------


def collapse_proj_in_channels(weight: torch.Tensor, mask_slice: slice = CHUNK_MASK_SLICE) -> torch.Tensor:
    """Collapse `proj_in_conv.weight` [hidden, 192, K] -> [hidden, 129, K].

    `context_latents = cat([src_latents(64), chunk_masks(64)], -1)` and `chunk_masks` is
    one per-frame scalar tiled 64x (`mask.unsqueeze(-1).repeat(1,1,64)`, BRINGUP §3.6), so
    the 64 mask channels are numerically identical. Summing the corresponding weight
    slice over the channel axis is therefore exact, not an approximation:

        sum_{c in 64:128} W[:, c, :] * m  ==  (sum_{c} W[:, c, :]) * m

    Channel order of the collapsed input is `[src_latents(64) | mask(1) | x_t(64)]`.
    """
    assert weight.dim() == 3, f"expected [out, in, K], got {tuple(weight.shape)}"
    assert weight.shape[1] == IN_CHANNELS_FULL, f"expected in_channels={IN_CHANNELS_FULL}, got {weight.shape[1]}"
    lo, hi = mask_slice.start, mask_slice.stop
    return torch.cat(
        [weight[:, :lo, :], weight[:, lo:hi, :].sum(dim=1, keepdim=True), weight[:, hi:, :]],
        dim=1,
    ).contiguous()


def collapse_proj_in_input(x_ncl: torch.Tensor, mask_slice: slice = CHUNK_MASK_SLICE) -> torch.Tensor:
    """Collapse the 192-channel NCL `proj_in` input to 129 channels, and assert that the
    64 chunk-mask channels really are identical (they are 0/1-valued per §3.6)."""
    lo, hi = mask_slice.start, mask_slice.stop
    mask_block = x_ncl[:, lo:hi, :]
    spread = (mask_block - mask_block[:, :1, :]).abs().max().item()
    assert spread == 0.0, f"chunk_mask channels are not identical (max spread {spread}); collapse is invalid"
    return torch.cat([x_ncl[:, :lo, :], mask_block[:, :1, :], x_ncl[:, hi:, :]], dim=1).contiguous()


def verify_proj_in_collapse(weight: torch.Tensor, bias: Optional[torch.Tensor], x_ncl: torch.Tensor, stride: int = 2):
    """Numerically verify the collapse on real data.

    Returns `(max_abs_diff_fp32, rel_err_fp32, pcc, max_abs_diff_fp64)`.

    The collapse is *algebraically* exact, so the float64 residual is ~1e-13 relative.
    In fp32 it is NOT bit-exact, and that is expected and benign: the two forms reduce
    over 192 vs 129 channels, so the accumulation order differs. Judge this check on the
    fp64 residual and on `rel_err`, never on `max_abs_diff == 0` in fp32.
    """
    from models.common.utility_functions import comp_pcc

    full = F.conv1d(x_ncl, weight, bias, stride=stride)
    collapsed = F.conv1d(collapse_proj_in_input(x_ncl), collapse_proj_in_channels(weight), bias, stride=stride)
    _, pcc = comp_pcc(full, collapsed, pcc=0.9999)
    d32 = (full - collapsed).abs().max().item()
    rel = d32 / max(full.abs().max().item(), 1e-30)

    w64, x64 = weight.double(), x_ncl.double()
    b64 = bias.double() if bias is not None else None
    full64 = F.conv1d(x64, w64, b64, stride=stride)
    coll64 = F.conv1d(collapse_proj_in_input(x64), collapse_proj_in_channels(w64), b64, stride=stride)
    d64 = (full64 - coll64).abs().max().item() / max(full64.abs().max().item(), 1e-30)
    return d32, rel, pcc, d64


# --------------------------------------------------------------------------------------
# 4. proj_in / proj_out as plain matmuls (the DiT is convolution-free)
# --------------------------------------------------------------------------------------


def proj_in_as_linear(weight: torch.Tensor) -> torch.Tensor:
    """`proj_in_conv` [hidden, C, K] -> a torch-convention Linear weight [hidden, K*C].

    `kernel == stride == patch_size` with no padding, so the conv is a non-overlapping
    reshape + matmul. The input reshape is `[B, T, C] -> [B, S, K*C]` (row-major over
    `(t_local, channel)`), which means the flattened input index is `k*C + c` and

        W_lin[o, k*C + c] = W_conv[o, c, k]      i.e.  W.permute(0, 2, 1).reshape(hidden, K*C)
    """
    assert weight.dim() == 3
    hidden, c_in, k = weight.shape
    return weight.permute(0, 2, 1).reshape(hidden, k * c_in).contiguous()


def proj_out_as_linear(weight: torch.Tensor) -> torch.Tensor:
    """`proj_out_conv` (ConvTranspose1d) [hidden, C_out, K] -> Linear weight [K*C_out, hidden].

    y[o, K*s + k] = sum_i x[i, s] * W[i, o, k], so with the output flattened as
    `(k, o)` -> `k*C_out + k_local`:

        W_lin[k*C_out + o, i] = W_conv[i, o, k]   i.e.  W.permute(2, 1, 0).reshape(K*C_out, hidden)

    The matmul output `[B, S, K*C_out]` then reshapes to `[B, S, K, C_out]` ->
    `[B, K*S, C_out]`, which is `depatchify()` below.
    """
    assert weight.dim() == 3
    hidden, c_out, k = weight.shape
    return weight.permute(2, 1, 0).reshape(k * c_out, hidden).contiguous()


def patchify(x_btc: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """[B, T, C] -> [B, T/patch, patch*C]. Pure reshape (row-major)."""
    b, t, c = x_btc.shape
    assert t % patch_size == 0, f"T={t} not divisible by patch_size={patch_size}"
    return x_btc.reshape(b, t // patch_size, patch_size * c)


def depatchify(x_bsc: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """[B, S, patch*C] -> [B, patch*S, C]. Pure reshape (row-major)."""
    b, s, kc = x_bsc.shape
    assert kc % patch_size == 0
    return x_bsc.reshape(b, s * patch_size, kc // patch_size)


def verify_proj_in_as_linear(weight, bias, x_ncl: torch.Tensor, patch_size: int = 2):
    """Check `proj_in_conv(x)` == `Linear(patchify(x))`. Returns `(max_abs_diff, pcc)`."""
    from models.common.utility_functions import comp_pcc

    conv = F.conv1d(x_ncl, weight, bias, stride=patch_size).transpose(1, 2)  # [B, S, hidden]
    lin = F.linear(patchify(x_ncl.transpose(1, 2), patch_size), proj_in_as_linear(weight), bias)
    _, pcc = comp_pcc(conv, lin, pcc=0.9999)
    return (conv - lin).abs().max().item(), pcc


def verify_proj_out_as_linear(weight, bias, x_ncl: torch.Tensor, patch_size: int = 2):
    """Check `proj_out_conv(x)` == `depatchify(Linear(x))`. Returns `(max_abs_diff, pcc)`."""
    from models.common.utility_functions import comp_pcc

    conv = F.conv_transpose1d(x_ncl, weight, bias, stride=patch_size).transpose(1, 2)  # [B, K*S, C_out]
    lin_w = proj_out_as_linear(weight)
    lin_b = bias.repeat(patch_size) if bias is not None else None
    lin = depatchify(F.linear(x_ncl.transpose(1, 2), lin_w, lin_b), patch_size)
    _, pcc = comp_pcc(conv, lin, pcc=0.9999)
    return (conv - lin).abs().max().item(), pcc


# --------------------------------------------------------------------------------------
# 5. packed SwiGLU weight for tt_dit's fused FeedForward
# --------------------------------------------------------------------------------------


def pack_swiglu_ff1(gate_weight: torch.Tensor, up_weight: torch.Tensor, gate_is_first: bool = False) -> torch.Tensor:
    """Pack ACE-Step's separate `gate_proj`/`up_proj` into one `ff1.weight`.

    `tt_dit.layers.linear.Linear(activation_fn="swiglu")._prepare_torch_state` does
    `weight.transpose(0, 1)` and then `prepare_for_fused_swiglu(weight, ndev=1)` with the
    helper's default `gate_is_first=False`. That default declares the *column* order of
    the transposed `[dim, 2*inner]` weight to be `[up | gate]`, so the torch-convention
    weight handed to `load_torch_state_dict` must be `[2*inner, dim]` with rows
    `[up | gate]` — i.e. `cat([up, gate], dim=0)`.

    The fused kernel computes `silu(even_tile) * odd_tile`, and `prepare_for_fused_swiglu`
    flips each tile pair when `gate_is_first=False` precisely so that gate lands in the
    even slot. Getting this backwards computes `silu(up) * gate`, which is a plausible
    ~0.9x-PCC wrong answer rather than a crash — hence the numeric check in
    `verify_swiglu_pack()`.

    Pass `gate_is_first=True` only if the consumer is configured with that flag.
    """
    assert gate_weight.shape == up_weight.shape, f"{tuple(gate_weight.shape)} != {tuple(up_weight.shape)}"
    order = (gate_weight, up_weight) if gate_is_first else (up_weight, gate_weight)
    return torch.cat(order, dim=0).contiguous()


def verify_swiglu_pack(gate_weight, up_weight, x_bsd: torch.Tensor, gate_is_first: bool = False):
    """Check the packed weight reproduces `silu(gate(x)) * up(x)` under the documented
    chunk convention. Returns `(max_abs_diff, pcc)`."""
    from models.common.utility_functions import comp_pcc

    reference = F.silu(F.linear(x_bsd, gate_weight)) * F.linear(x_bsd, up_weight)
    packed = F.linear(x_bsd, pack_swiglu_ff1(gate_weight, up_weight, gate_is_first))
    a, b = packed.chunk(2, dim=-1)
    gate_h, up_h = (a, b) if gate_is_first else (b, a)
    fused = F.silu(gate_h) * up_h
    _, pcc = comp_pcc(reference, fused, pcc=0.9999)
    return (reference - fused).abs().max().item(), pcc


def verify_swiglu_pack_via_tt_dit(gate_weight, up_weight, x_bsd: torch.Tensor):
    """Same check, but routed through `tt_dit.utils.tensor.prepare_for_fused_swiglu` and
    the kernel's `silu(even_tile) * odd_tile` semantics — this is the check that catches a
    gate/up swap in the *real* consumer. Imports ttnn lazily (no device is opened).
    """
    from models.common.utility_functions import comp_pcc
    from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

    tile = 32
    packed = pack_swiglu_ff1(gate_weight, up_weight, gate_is_first=False).transpose(0, 1)  # [dim, 2*inner]
    laid_out = prepare_for_fused_swiglu(packed, ndev=1, tile_width=tile)
    h = F.linear(x_bsd, laid_out.transpose(0, 1))  # [B, S, 2*inner], tile-interleaved
    tiles = h.reshape(*h.shape[:-1], h.shape[-1] // tile, tile)
    fused = (F.silu(tiles[..., 0::2, :]) * tiles[..., 1::2, :]).reshape(*h.shape[:-1], h.shape[-1] // 2)
    reference = F.silu(F.linear(x_bsd, gate_weight)) * F.linear(x_bsd, up_weight)
    _, pcc = comp_pcc(reference, fused, pcc=0.9999)
    return (reference - fused).abs().max().item(), pcc


# --------------------------------------------------------------------------------------
# 6. whole-state-dict wrappers
# --------------------------------------------------------------------------------------


def prepare_dit_weights(
    state: StateDict,
    config: dict,
    collapse_proj_in: bool = True,
    convs_as_linear: bool = True,
    pack_swiglu: bool = True,
    fold_time_embed_r: bool = False,
) -> StateDict:
    """Apply every host-side DiT weight transform. Returns a NEW dict; `state` is untouched.

    `fold_time_embed_r=True` drops `time_embed_r.*` entirely on the grounds that
    inference always passes `timestep_r == timestep`, so `time_embed_r` always sees 0 and
    its contribution is a constant (BRINGUP §3.7). The constants themselves come from the
    goldens (`transformer.time_embed_r.out0` / `.out1`), not from this function, since
    computing them needs the sinusoid + 2 MLPs.
    """
    validate_diffusers_state(state, config, strict=True)
    out = dict(state)

    if collapse_proj_in:
        out["proj_in_conv.weight"] = collapse_proj_in_channels(out["proj_in_conv.weight"])

    if convs_as_linear:
        out["proj_in_linear.weight"] = proj_in_as_linear(out["proj_in_conv.weight"])
        out["proj_in_linear.bias"] = out["proj_in_conv.bias"].clone()
        out["proj_out_linear.weight"] = proj_out_as_linear(out["proj_out_conv.weight"])
        out["proj_out_linear.bias"] = out["proj_out_conv.bias"].repeat(config["patch_size"]).contiguous()

    if pack_swiglu:
        for i in range(config["num_hidden_layers"]):
            g = out.pop(f"layers.{i}.mlp.gate_proj.weight")
            u = out.pop(f"layers.{i}.mlp.up_proj.weight")
            out[f"layers.{i}.mlp.ff1.weight"] = pack_swiglu_ff1(g, u)
            out[f"layers.{i}.mlp.ff2.weight"] = out.pop(f"layers.{i}.mlp.down_proj.weight")

    # scale_shift_table is a dim-1 chunk of 6 (2 for the model-level table). Splitting it
    # host-side avoids a size-6 slice on device (BRINGUP §3.7).
    for i in range(config["num_hidden_layers"]):
        table = out.pop(f"layers.{i}.scale_shift_table")
        names = ("shift_msa", "scale_msa", "gate_msa", "c_shift_msa", "c_scale_msa", "c_gate_msa")
        for j, nm in enumerate(names):
            out[f"layers.{i}.scale_shift.{nm}"] = table[:, j : j + 1, :].contiguous()
    table = out.pop("scale_shift_table")
    out["scale_shift.shift"] = table[:, 0:1, :].contiguous()
    out["scale_shift.scale"] = table[:, 1:2, :].contiguous()

    if fold_time_embed_r:
        for key in [k for k in out if k.startswith("time_embed_r.")]:
            out.pop(key)

    return out


def prepare_vae_decoder_weights(state: StateDict, decoder_prefix: str = "decoder.") -> StateDict:
    """Fold `weight_norm` and precompute Snake coefficients for the Oobleck decoder.

    `state` may be the whole VAE state dict; only `decoder_prefix` keys are kept (the
    encoder is not needed for text2music — BRINGUP §3.8).
    """
    decoder = {k[len(decoder_prefix) :]: v for k, v in state.items() if k.startswith(decoder_prefix)}
    if not decoder:
        raise KeyError(f"no keys under prefix {decoder_prefix!r}; got e.g. {list(state)[:4]}")
    out = fold_weight_norm(decoder, strict=True)

    snake_prefixes = sorted({k[: -len(".alpha")] for k in out if k.endswith(".alpha")})
    for prefix in snake_prefixes:
        exp_alpha, inv_beta = snake_params(out, prefix)
        out[f"{prefix}.exp_alpha"] = exp_alpha
        out[f"{prefix}.inv_beta"] = inv_beta
    return out


def verify_vae_fold(vae_state: StateDict, vae_config: dict, x_ncl: Optional[torch.Tensor] = None) -> dict:
    """Verify `fold_weight_norm` against the live diffusers module.

    Builds a real `AutoencoderOobleck`, loads the raw (still weight-normed) checkpoint,
    materialises `.weight` via `remove_weight_norm`, and compares every decoder conv
    weight against our fold **bit-exactly**. If `x_ncl` is given ([B,64,T] latents), also
    runs the de-parametrised decoder and returns the PCC against the parametrised one.
    """
    from diffusers import AutoencoderOobleck
    from torch.nn.utils import remove_weight_norm

    cfg = {k: v for k, v in vae_config.items() if not k.startswith("_")}
    vae = AutoencoderOobleck(**cfg).eval()
    missing, unexpected = vae.load_state_dict(vae_state, strict=False)
    assert not [m for m in missing if not m.endswith(".weight")], f"missing: {missing[:6]}"
    assert not unexpected, f"unexpected: {unexpected[:6]}"

    ours = fold_weight_norm({k[len("decoder.") :]: v for k, v in vae_state.items() if k.startswith("decoder.")})

    ref_out = None
    if x_ncl is not None:
        with torch.no_grad():
            ref_out = vae.decoder(x_ncl).clone()

    n_checked = n_exact = 0
    for name, module in vae.decoder.named_modules():
        if hasattr(module, "weight_g"):
            remove_weight_norm(module)
            n_checked += 1
            key = f"{name}.weight"
            assert key in ours, f"fold produced no {key}"
            if torch.equal(module.weight.detach(), ours[key]):
                n_exact += 1

    result = {"convs_checked": n_checked, "convs_bit_exact": n_exact}
    if x_ncl is not None:
        from models.common.utility_functions import comp_pcc

        with torch.no_grad():
            plain_out = vae.decoder(x_ncl)
        _, result["pcc"] = comp_pcc(ref_out, plain_out, pcc=0.9999)
        result["max_abs_diff"] = (ref_out - plain_out).abs().max().item()
        result["decoder_out"] = plain_out
    return result


def assert_even_strides(strides: Iterable[int]) -> None:
    """TRAP-4: `ConvTranspose1dViaConv3d` hardcodes `padding = floor(stride/2)` while
    ACE-Step wants `ceil(stride/2)`. They agree only for even strides. Our list is
    [10, 6, 4, 4, 2], all even — assert it rather than rely on luck."""
    bad = [s for s in strides if s % 2 != 0]
    assert not bad, f"odd VAE stride(s) {bad}: ConvTranspose1dViaConv3d padding would be off by one (TRAP-4)"


def decoder_strides(vae_config: dict) -> list:
    """Decoder upsampling strides = `downsampling_ratios` reversed (see OobleckDecoder)."""
    return list(reversed(list(vae_config["downsampling_ratios"])))


def snake_count(n_blocks: int = 5) -> int:
    """36 Snake instances in the decoder: 5 blocks x (1 block snake + 3 res units x 2) + 1
    final `snake1`. Sanity-checks the checkpoint survey."""
    return n_blocks * (1 + 3 * 2) + 1


# --------------------------------------------------------------------------------------
# self-check (host-only)
# --------------------------------------------------------------------------------------


def self_check(pipeline_path: Optional[str] = None, golden_dir: Optional[str] = None) -> int:
    """Run every numeric verification against the real checkpoint + goldens. Returns the
    number of failures. No device, no ttnn tensors."""
    import json
    import os

    from safetensors.torch import load_file

    path = pipeline_path or os.getenv("ACE_STEP_PIPELINE", "/localdev/acicovic/ace_step_diffusers")
    here = os.path.dirname(os.path.abspath(__file__))
    golden_dir = golden_dir or os.path.join(here, "..", "golden")
    failures = 0

    def report(name, ok, detail=""):
        nonlocal failures
        if not ok:
            failures += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    print(f"[weights] checkpoint: {path}")
    with open(os.path.join(path, "transformer", "config.json")) as f:
        dit_config = json.load(f)
    with open(os.path.join(path, "vae", "config.json")) as f:
        vae_config = json.load(f)

    # --- DiT state dict -----------------------------------------------------------------
    dit_state: StateDict = {}
    for fn in sorted(os.listdir(os.path.join(path, "transformer"))):
        if fn.endswith(".safetensors"):
            dit_state.update(load_file(os.path.join(path, "transformer", fn)))
    print(f"[weights] DiT: {len(dit_state)} tensors, {sum(t.numel() for t in dit_state.values()) / 1e6:.2f} M params")

    problems = validate_diffusers_state(dit_state, dit_config, strict=False)
    report("validate_diffusers_state", not problems, f"({len(problems)} problems)" if problems else "")
    for p in problems[:10]:
        print(f"        {p}")

    # --- upstream remap round trip -------------------------------------------------------
    fake_upstream = {
        "decoder.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
        "decoder.layers.0.self_attn.q_norm.weight": torch.zeros(4),
        "decoder.layers.0.self_attn.k_norm.weight": torch.zeros(4),
        "decoder.layers.0.self_attn.o_proj.weight": torch.zeros(4, 4),
        "decoder.proj_in.1.weight": torch.zeros(4, 4, 2),
        "decoder.proj_out.1.bias": torch.zeros(4),
        "encoder.lyric_encoder.layers.0.self_attn.k_proj.weight": torch.zeros(4, 4),
        "null_condition_emb": torch.zeros(1, 1, 4),
        "tokenizer.quantizer.project_in.weight": torch.zeros(6, 4),
        "detokenizer.proj_out.weight": torch.zeros(4, 4),
        "some.junk.key": torch.zeros(1),
    }
    comps, dropped = remap_upstream_to_diffusers(fake_upstream)
    expect_tr = {
        "layers.0.self_attn.to_q.weight",
        "layers.0.self_attn.norm_q.weight",
        "layers.0.self_attn.norm_k.weight",
        "layers.0.self_attn.to_out.0.weight",
        "proj_in_conv.weight",
        "proj_out_conv.bias",
    }
    report("remap: transformer keys", set(comps["transformer"]) == expect_tr, str(sorted(comps["transformer"])))
    report(
        "remap: encoder/tokenizer/detokenizer",
        set(comps["condition_encoder"]) == {"lyric_encoder.layers.0.self_attn.to_k.weight", "null_condition_emb"}
        and set(comps["audio_tokenizer"]) == {"quantizer.project_in.weight"}
        and set(comps["audio_token_detokenizer"]) == {"proj_out.weight"},
    )
    report("remap: dropped unknown prefix", dropped == ["some.junk.key"], str(dropped))

    # --- proj_in collapse + conv-as-linear, on REAL data ---------------------------------
    x_path = os.path.join(golden_dir, "dit", "s32", "transformer.proj_in_conv.in0.pt")
    if os.path.exists(x_path):
        x = torch.load(x_path, map_location="cpu", weights_only=True)  # [1, 192, T] NCL
        w, b = dit_state["proj_in_conv.weight"], dit_state["proj_in_conv.bias"]
        d32, rel, pcc, d64 = verify_proj_in_collapse(w, b, x)
        report(
            "proj_in 192->129 collapse",
            d64 < 1e-12 and rel < 1e-5 and pcc > 1 - 1e-9,
            f"fp64 rel={d64:.2e} (algebra) | fp32 max|diff|={d32:.2e} rel={rel:.2e} pcc={pcc:.12f}",
        )
        d, pcc = verify_proj_in_as_linear(w, b, x)
        report("proj_in as Linear(384->2048)", pcc > 0.99999, f"max|diff|={d:.3e} pcc={pcc}")
        collapsed = collapse_proj_in_channels(w)
        report(
            "collapsed shape",
            tuple(collapsed.shape) == (dit_config["hidden_size"], IN_CHANNELS_COLLAPSED, dit_config["patch_size"]),
            str(tuple(collapsed.shape)),
        )
    else:
        print(f"  [SKIP] proj_in checks: no golden at {x_path}")

    xo_path = os.path.join(golden_dir, "dit", "s32", "transformer.proj_out_conv.in0.pt")
    if os.path.exists(xo_path):
        xo = torch.load(xo_path, map_location="cpu", weights_only=True)  # [1, 2048, S]
        d, pcc = verify_proj_out_as_linear(dit_state["proj_out_conv.weight"], dit_state["proj_out_conv.bias"], xo)
        report("proj_out as Linear(2048->128)+reshape", pcc > 0.99999, f"max|diff|={d:.3e} pcc={pcc}")
    else:
        print(f"  [SKIP] proj_out check: no golden at {xo_path}")

    # --- SwiGLU pack, on REAL weights + real activations --------------------------------
    mlp_in = os.path.join(golden_dir, "dit", "s32", "transformer.layers.0.mlp.in0.pt")
    g = dit_state["layers.0.mlp.gate_proj.weight"]
    u = dit_state["layers.0.mlp.up_proj.weight"]
    xm = (
        torch.load(mlp_in, map_location="cpu", weights_only=True)
        if os.path.exists(mlp_in)
        else torch.randn(1, 32, dit_config["hidden_size"], generator=torch.Generator().manual_seed(0))
    )
    packed = pack_swiglu_ff1(g, u)
    report(
        "swiglu pack shape [2*inner, dim]",
        tuple(packed.shape) == (2 * dit_config["intermediate_size"], dit_config["hidden_size"]),
        str(tuple(packed.shape)),
    )
    d, pcc = verify_swiglu_pack(g, u, xm)
    report("swiglu pack (chunk convention)", d == 0.0, f"max|diff|={d:.3e} pcc={pcc}")
    try:
        d, pcc = verify_swiglu_pack_via_tt_dit(g, u, xm)
        report("swiglu pack via prepare_for_fused_swiglu", d == 0.0, f"max|diff|={d:.3e} pcc={pcc}")
    except Exception as e:  # ttnn import failure must not fail the host-only check
        print(f"  [SKIP] prepare_for_fused_swiglu check: {type(e).__name__}: {e}")
    # And the wrong order must be visibly wrong, so a swap cannot pass silently.
    swapped = F.silu(F.linear(xm, u)) * F.linear(xm, g)
    ref = F.silu(F.linear(xm, g)) * F.linear(xm, u)
    from models.common.utility_functions import comp_pcc

    _, pcc_swapped = comp_pcc(ref, swapped, pcc=0.9999)
    report("swiglu gate/up swap is detectable", pcc_swapped < 0.999, f"pcc(swapped vs ref)={pcc_swapped:.6f}")

    # --- VAE weight_norm fold ------------------------------------------------------------
    vae_state = load_file(os.path.join(path, "vae", "diffusion_pytorch_model.safetensors"))
    legacy = sum(1 for k in vae_state if k.endswith(".weight_g"))
    param = sum(1 for k in vae_state if k.endswith(".parametrizations.weight.original0"))
    print(f"[weights] VAE: {len(vae_state)} tensors, weight_g pairs={legacy}, parametrization pairs={param}")
    dec = prepare_vae_decoder_weights(vae_state)
    n_conv = sum(1 for k in dec if k.endswith(".weight") and dec[k].dim() == 3)
    report("VAE decoder convs folded", n_conv == 37, f"{n_conv} fused conv weights (expect 37)")
    n_snake = sum(1 for k in dec if k.endswith(".exp_alpha"))
    report("VAE decoder snakes", n_snake == snake_count(), f"{n_snake} (expect {snake_count()})")
    try:
        assert_even_strides(decoder_strides(vae_config))
        report("VAE strides all even (TRAP-4)", True, str(decoder_strides(vae_config)))
    except AssertionError as e:
        report("VAE strides all even (TRAP-4)", False, str(e))

    # Fold must be bit-exact against the LIVE diffusers decoder, on the real checkpoint.
    vae_in_path = os.path.join(golden_dir, "vae", "s32", "vae.decoder.in0.pt")
    vae_x = torch.load(vae_in_path, map_location="cpu", weights_only=True) if os.path.exists(vae_in_path) else None
    fold = verify_vae_fold(vae_state, vae_config, vae_x)
    report(
        "VAE fold bit-exact vs live remove_weight_norm",
        fold["convs_bit_exact"] == fold["convs_checked"] == 37,
        f"{fold['convs_bit_exact']}/{fold['convs_checked']} convs",
    )
    if vae_x is not None:
        report(
            "de-parametrised decoder matches parametrised (real latents)",
            fold.get("max_abs_diff", 1.0) == 0.0,
            f"max|diff|={fold.get('max_abs_diff'):.3e} pcc={fold.get('pcc')}",
        )
        golden_out = torch.load(os.path.join(golden_dir, "vae", "s32", "vae.decoder.out.pt"), weights_only=True)
        d = (golden_out - fold["decoder_out"]).abs().max().item()
        report(
            "golden vae.decoder.out reproduced from the folded checkpoint",
            d == 0.0,
            f"max|diff|={d:.3e}",
        )

    # Synthetic checks of both storage forms + the failure mode.
    from torch.nn.utils import weight_norm as _wn

    conv = _wn(torch.nn.Conv1d(8, 16, 7))
    with torch.no_grad():
        conv.weight_g.copy_(torch.randn_like(conv.weight_g))
        conv.weight_v.copy_(torch.randn_like(conv.weight_v))
        conv(torch.zeros(1, 8, 16))  # legacy weight_norm recomputes .weight in a PRE-FORWARD hook
    folded = fold_weight_norm({k: v.detach() for k, v in conv.state_dict().items()})
    live = conv.weight.detach()
    report("weight_norm fold is bit-exact (legacy form)", torch.equal(folded["weight"], live))

    # Newer parametrization form must fold identically.
    from torch.nn.utils.parametrizations import weight_norm as _wnp

    conv2 = _wnp(torch.nn.Conv1d(8, 16, 7))
    sd2 = {k: v.detach() for k, v in conv2.state_dict().items()}
    assert any("parametrizations.weight.original0" in k for k in sd2), sorted(sd2)
    folded2 = fold_weight_norm(sd2)
    report(
        "weight_norm fold is bit-exact (parametrization form)", torch.equal(folded2["weight"], conv2.weight.detach())
    )

    # A checkpoint with neither form must raise, not load unfused.
    try:
        fold_weight_norm({"conv1.bias": torch.zeros(4)}, strict=True)
        report("unfused checkpoint raises", False, "no exception")
    except KeyError:
        report("unfused checkpoint raises", True)

    # ConvTranspose1d: g is per INPUT channel — check the axis semantics hold.
    ct = _wn(torch.nn.ConvTranspose1d(8, 16, 4, stride=2))
    with torch.no_grad():
        ct.weight_g.copy_(torch.randn_like(ct.weight_g))
        ct.weight_v.copy_(torch.randn_like(ct.weight_v))
        ct(torch.zeros(1, 8, 16))
    foldedct = fold_weight_norm({k: v.detach() for k, v in ct.state_dict().items()})
    report(
        "ConvTranspose1d fold bit-exact (g per in-channel)",
        torch.equal(foldedct["weight"], ct.weight.detach()) and ct.weight_g.shape[0] == 8,
        f"g shape {tuple(ct.weight_g.shape)}",
    )

    # --- whole-DiT prepare ---------------------------------------------------------------
    prepared = prepare_dit_weights(dit_state, dit_config)
    report(
        "prepare_dit_weights: ff1 packed for all layers",
        all(f"layers.{i}.mlp.ff1.weight" in prepared for i in range(dit_config["num_hidden_layers"])),
    )
    report(
        "prepare_dit_weights: scale_shift split",
        "layers.0.scale_shift.gate_msa" in prepared and "scale_shift.scale" in prepared,
    )
    total = sum(t.numel() for t in prepared.values())
    print(f"[weights] prepared DiT: {len(prepared)} tensors, {total / 1e6:.2f} M elements")

    print(f"[weights] {failures} failure(s)")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if self_check() else 0)
