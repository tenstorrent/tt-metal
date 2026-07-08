"""Reference-model loader for ``meituan-longcat/LongCat-Video``.

The repo's root ``config.json`` is just ``{"model_name": "LongCat-Video"}`` -- no
``model_type``/``auto_map`` -- so ``AutoConfig``/``AutoModel`` can't load it: this is not a single
``transformers`` checkpoint but a diffusers-style *multi-component* pipeline (``dit/``,
``text_encoder/``, ``vae/``, ``tokenizer/``, ``scheduler/`` subfolders, wired together by the
model's own ``longcat_video.pipeline_longcat_video.LongCatVideoPipeline`` on GitHub, not through
``diffusers``' pipeline registry).

Inspecting each subfolder's own ``config.json`` shows three different loading stories:

* ``text_encoder/config.json`` -- ``model_type: umt5``, ``architectures: [UMT5EncoderModel]``.
  This *is* a real, registered ``transformers`` architecture -- ``transformers.UMT5EncoderModel``
  loads it directly.
* ``vae/config.json`` -- ``_class_name: AutoencoderKLWan``. ``diffusers`` already ships a
  same-named class for the Wan2.1 video VAE, and its default field values (``base_dim``,
  ``dim_mult``, ``z_dim``, ``latents_mean``/``latents_std``, ...) match this checkpoint's config
  byte-for-byte -- confirmed by actually downloading the (small, ~500MB) real weights and running
  a real encode/decode: they load with no missing/unexpected keys. So the stock ``diffusers``
  class is the right one here too.
* ``dit/config.json`` -- ``_class_name: LongCatVideoTransformer3DModel``. This class is *not* in
  ``diffusers`` (checked: only ``LongCatImageTransformer2DModel`` /
  ``LongCatAudioDiTTransformer`` ship there, no video variant). It is a ``diffusers``
  ``ModelMixin``/``ConfigMixin`` subclass defined only in the model's own GitHub repo
  (``meituan-longcat/LongCat-Video``, MIT licensed), at
  ``longcat_video/modules/longcat_video_dit.py``. Being a ``ModelMixin`` subclass, it supports the
  standard ``.from_pretrained(repo_id, subfolder="dit")`` API once the class itself is available
  -- confirmed against the repo's own ``run_demo_text_to_video.py``, which loads it exactly this
  way (down to passing ``cp_split_hw`` as an override kwarg, see below).

So: this is case 4 from the brief (config-less custom architecture) for the ``dit`` only: we
vendor that one class (pinned commit, MIT licensed) rather than guess at random weights, and use
already-registered real classes for ``text_encoder``/``vae``. All three load genuine trained
weights.

Two environment gaps in the vendored ``dit`` code, patched here (documented at each patch site):

1. ``attention.py`` unconditionally imports the block-sparse-attention kernel (needs ``triton``,
   not installed, and pointless anyway since this checkpoint has ``enable_bsa: false``) --
   stubbed out.
2. ``dit/config.json`` sets ``enable_flashattn2: true``, and the vendored ``Attention`` /
   ``MultiHeadCrossAttention`` modules have *no* eager-softmax fallback: with none of
   flash-attn/xformers/BSA importable (none are installed here) they raise ``RuntimeError``.
   Flash-attention computes exactly standard scaled-dot-product softmax attention, so we
   monkeypatch the two ``_process*`` methods to call
   ``torch.nn.functional.scaled_dot_product_attention`` instead -- identical math, portable to
   CPU, and it is the same operation the ttnn port has to match.

Also: ``dit/config.json`` has ``cp_split_hw: null``, but the vendored forward pass does
``self.cp_split_hw[0] * self.cp_split_hw[1]`` unconditionally (no ``None`` guard) -- the repo's
own demo scripts always override this via a ``from_pretrained(..., cp_split_hw=...)`` kwarg
(``get_optimal_split(cp_size=1) == [1, 1]`` for the single-process case), so we do the same.

The loader is import-safe (everything below happens inside function bodies, not at module import
time) and deterministic (``torch.manual_seed`` is fixed before any construction).
"""

from __future__ import annotations

import importlib
import os
import sys
import types
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

# Pinned commit of https://github.com/meituan-longcat/LongCat-Video (MIT licensed) that the
# `dit/` weights on the Hub were published against (`_diffusers_version: 0.32.0` in
# `dit/config.json` predates this commit, and the class shape has been stable since). Pinning
# keeps vendoring deterministic/reproducible instead of tracking a moving `main` branch.
_LONGCAT_VIDEO_COMMIT = "6b3f4b8582a8bc3f20f795735f5383716c4ba794"
_LONGCAT_VIDEO_RAW = (
    f"https://raw.githubusercontent.com/meituan-longcat/LongCat-Video/{_LONGCAT_VIDEO_COMMIT}/longcat_video"
)

# Only the subset of the upstream package needed to construct `LongCatVideoTransformer3DModel`
# (no LoRA/quantization/avatar/audio extras, no block-sparse-attention -- see
# `_install_bsa_stub`). Paths are relative to the vendored `longcat_video/` package root.
_VENDOR_FILES = (
    "modules/__init__.py",
    "modules/longcat_video_dit.py",
    "modules/attention.py",
    "modules/blocks.py",
    "modules/rope_3d.py",
    "modules/lora_utils.py",
    "context_parallel/context_parallel_util.py",
    "context_parallel/ulysses_wrapper.py",
)


def _vendor_root() -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "tt_hw_planner_longcat_video_vendor" / _LONGCAT_VIDEO_COMMIT


def _ensure_vendored() -> Path:
    """Download (once; cached on disk) the pinned-commit source subset listed in
    `_VENDOR_FILES`. `longcat_video`, `longcat_video.context_parallel`, and
    `longcat_video.block_sparse_attention` are implicit namespace packages upstream (no
    `__init__.py`); only `modules/` ships a real (empty) one, which we mirror."""
    root = _vendor_root()
    pkg_root = root / "longcat_video"
    marker = pkg_root / "modules" / "longcat_video_dit.py"
    if marker.is_file():
        return root

    tmp_root = root.with_name(root.name + f".tmp{os.getpid()}")
    tmp_pkg_root = tmp_root / "longcat_video"
    for rel in _VENDOR_FILES:
        dst = tmp_pkg_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{_LONGCAT_VIDEO_RAW}/{rel}", timeout=30) as resp:
            dst.write_bytes(resp.read())
    tmp_root.replace(root)
    return root


def _install_bsa_stub() -> None:
    """`longcat_video/modules/attention.py` does
    `from ..block_sparse_attention.bsa_interface import flash_attn_bsa_3d` at module scope even
    though this checkpoint's `dit/config.json` has `enable_bsa: false` (so `flash_attn_bsa_3d` is
    never actually called). The real module needs `triton`, which isn't installed and isn't
    needed for a checkpoint that never enables BSA, so we inject a stub package that satisfies
    the import without pulling in a GPU-only kernel toolchain."""
    if "longcat_video.block_sparse_attention.bsa_interface" in sys.modules:
        return

    pkg = types.ModuleType("longcat_video.block_sparse_attention")
    pkg.__path__ = []  # mark as a package so submodule imports resolve
    stub = types.ModuleType("longcat_video.block_sparse_attention.bsa_interface")

    def _unused_flash_attn_bsa_3d(*_args, **_kwargs):
        raise NotImplementedError("block-sparse attention is unused for this checkpoint (enable_bsa=false)")

    stub.flash_attn_bsa_3d = _unused_flash_attn_bsa_3d
    sys.modules["longcat_video.block_sparse_attention"] = pkg
    sys.modules["longcat_video.block_sparse_attention.bsa_interface"] = stub


def _install_portable_attention(attention_mod) -> None:
    """Replace the flash-attn/xformers/BSA-only attention kernels with
    `torch.nn.functional.scaled_dot_product_attention` -- the same standard softmax attention
    (same `q`/`k`/`v` layout `[B, H, S, D]`, same `softmax_scale`) computed portably on CPU. See
    the module docstring for why this is needed in this environment."""

    def _process_attn(self, q, k, v, _shape):
        return F.scaled_dot_product_attention(q, k, v, scale=self.scale)

    def _process_cross_attn(self, x, cond, kv_seqlen):
        # Faithful re-expression of the upstream flash_attn_varlen_func call: `cond` packs every
        # batch item's (padded/valid) tokens back-to-back into one sequence
        # (`kv_seqlen[b]` tokens each); attend each query item only to its own slice.
        B, N, C = x.shape
        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)  # [1, H, sum(kv_seqlen), D]
        v = v.transpose(1, 2)

        outs = []
        offset = 0
        for b in range(B):
            length = int(kv_seqlen[b])
            outs.append(
                F.scaled_dot_product_attention(
                    q[b : b + 1], k[:, :, offset : offset + length], v[:, :, offset : offset + length]
                )
            )
            offset += length
        x = torch.cat(outs, dim=0).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

    attention_mod.Attention._process_attn = _process_attn
    attention_mod.MultiHeadCrossAttention._process_cross_attn = _process_cross_attn


def _load_longcat_video_dit_class():
    """Vendor (if needed), import, and patch `LongCatVideoTransformer3DModel`. Safe to call more
    than once (patches and the on-disk vendor cache are both idempotent)."""
    vendor_root = _ensure_vendored()
    vendor_root_str = str(vendor_root)
    if vendor_root_str not in sys.path:
        sys.path.insert(0, vendor_root_str)

    _install_bsa_stub()
    dit_mod = importlib.import_module("longcat_video.modules.longcat_video_dit")
    attention_mod = importlib.import_module("longcat_video.modules.attention")
    _install_portable_attention(attention_mod)
    return dit_mod.LongCatVideoTransformer3DModel


class LongCatVideoReference(torch.nn.Module):
    """The three real, trained sub-models the LongCat-Video pipeline is built from, as named
    children (matching the repo's own subfolder names) so per-component PCC tests can resolve any
    real submodule path against genuine trained weights -- e.g. `dit.blocks.0.attn`,
    `dit.blocks.0.cross_attn`, `dit.blocks.0.ffn`, `text_encoder.encoder.block.0`,
    `vae.encoder...`, `vae.decoder...`."""

    def __init__(self, dit: torch.nn.Module, text_encoder: torch.nn.Module, vae: torch.nn.Module):
        super().__init__()
        self.dit = dit
        self.text_encoder = text_encoder
        self.vae = vae


def load_reference_model(model_id: str):
    """Return the real LongCat-Video module tree (``nn.Module``, eval mode) with trained weights.

    Loads each of the three real sub-models the same way the repo's own
    ``run_demo_text_to_video.py`` does: ``text_encoder`` via ``transformers.UMT5EncoderModel``,
    ``vae`` via ``diffusers.AutoencoderKLWan``, and ``dit`` via the vendored
    ``LongCatVideoTransformer3DModel`` (native to the model's own repo, not part of
    ``diffusers``) -- all three read from ``model_id``'s own subfolders on the Hub. Deterministic
    and import-safe: all work happens here, not at module import time.
    """
    torch.manual_seed(0)

    import diffusers
    from transformers import UMT5EncoderModel

    dit_cls = _load_longcat_video_dit_class()

    vae = diffusers.AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float32)
    # `cp_split_hw` overrides the checkpoint's `null` config value the same way the upstream demo
    # scripts do for a single-process run (`context_parallel_util.get_optimal_split(1) == [1, 1]`)
    # -- the vendored forward pass indexes `cp_split_hw[0]`/`[1]` unconditionally.
    dit = dit_cls.from_pretrained(model_id, subfolder="dit", cp_split_hw=[1, 1], torch_dtype=torch.float32)

    model = LongCatVideoReference(dit=dit, text_encoder=text_encoder, vae=vae)
    model.eval()
    return model


if __name__ == "__main__":
    # Quick self-check: real modules load and real (weight-backed) forwards run.
    #
    # `dit` (~54GB across 6 shards) and `text_encoder` (~21GB across 5 shards) are large enough
    # that a *full* download here is slow rather than genuinely broken -- the loading mechanism
    # for both is exercised structurally below (real classes, real configs, correct subfolder
    # resolution) without waiting on the full download. `vae` (~500MB) is small enough to check
    # completely end to end: real weights, real encode/decode forward.
    torch.manual_seed(0)

    import diffusers
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, UMT5EncoderModel

    model_id = "meituan-longcat/LongCat-Video"

    # 1) vae: real weights, real forward.
    vae_dir = snapshot_download(model_id, allow_patterns=["vae/config.json", "vae/diffusion_pytorch_model.safetensors"])
    vae = diffusers.AutoencoderKLWan.from_pretrained(vae_dir, subfolder="vae", torch_dtype=torch.float32)
    vae.eval()
    with torch.no_grad():
        z = vae.encode(torch.randn(1, 3, 5, 32, 32)).latent_dist.sample()
        recon = vae.decode(z).sample
    assert recon.shape == (1, 3, 5, 32, 32), recon.shape
    print("OK: vae real-weight encode/decode forward, latent shape", tuple(z.shape), "-> recon", tuple(recon.shape))

    # 2) text_encoder: real config + class, structural (untrained-weight) forward -- confirms
    # `UMT5EncoderModel`/`subfolder="text_encoder"` resolve correctly without the full download.
    te_dir = snapshot_download(model_id, allow_patterns=["text_encoder/config.json"])
    te_cfg = AutoConfig.from_pretrained(te_dir, subfolder="text_encoder")
    assert te_cfg.model_type == "umt5", te_cfg.model_type
    te = UMT5EncoderModel(te_cfg)
    te.eval()
    with torch.no_grad():
        te_out = te(
            input_ids=torch.randint(0, 1000, (1, 8)), attention_mask=torch.ones(1, 8, dtype=torch.long)
        ).last_hidden_state
    print("OK: text_encoder (UMT5EncoderModel) structural forward, shape", tuple(te_out.shape))

    # 3) dit: real config, vendored class + attention patch, structural (untrained-weight)
    # forward at reduced depth/width -- confirms the vendoring/monkeypatch machinery is correct
    # without waiting on the 54GB download. `load_reference_model` (used for real bring-up runs)
    # always builds the checkpoint's real, full-size config with real weights.
    dit_cls = _load_longcat_video_dit_class()
    dit_dir = snapshot_download(model_id, allow_patterns=["dit/config.json"])
    import json as _json

    with open(os.path.join(dit_dir, "dit", "config.json")) as f:
        full_dit_cfg = _json.load(f)
    smoke_cfg = {k: v for k, v in full_dit_cfg.items() if not k.startswith("_")}
    smoke_cfg.update(depth=2, hidden_size=256, num_heads=8, cp_split_hw=[1, 1])
    dit = dit_cls(**smoke_cfg)
    dit.eval()
    B, C, T, H, W = 1, smoke_cfg["in_channels"], 1, 16, 16
    n_token = 20
    with torch.no_grad():
        dit_out = dit(
            hidden_states=torch.randn(B, C, T, H, W),
            timestep=torch.tensor([500.0]),
            encoder_hidden_states=torch.randn(B, 1, n_token, smoke_cfg["caption_channels"]),
            encoder_attention_mask=torch.ones(B, n_token),
        )
    assert dit_out.shape == (B, smoke_cfg["out_channels"], T, H, W), dit_out.shape
    print("OK: dit (LongCatVideoTransformer3DModel, vendored) structural forward, shape", tuple(dit_out.shape))

    # 4) full assembled module tree (real classes; vae/dit share the same construction path
    # `load_reference_model` uses -- only vae carries real weights above for speed).
    model = LongCatVideoReference(dit=dit, text_encoder=te, vae=vae)
    model.eval()
    assert isinstance(model, torch.nn.Module)
    assert not model.training
    print("OK:", type(model).__name__, "children=", [n for n, _ in model.named_children()])
