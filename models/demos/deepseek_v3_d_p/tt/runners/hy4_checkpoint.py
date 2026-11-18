# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview checkpoint reader for the prefill weight-cache rebuild (``tt/runners/weight_categories.py``).

The cache rebuild reads the DeepSeek-V3 HF key layout. Hy4's checkpoints differ in four ways, all
handled here so the rest of the rebuild (category checks, the lazy per-layer reader, the model's own
cache writers) is shared with GLM-5.3 unchanged:

  * routed experts are FUSED per layer: ``mlp.experts.gate_up_proj`` ``[E, 2 * I, H]`` (gate rows
    first, then up, per expert) and ``mlp.experts.down_proj`` ``[E, H, I]``, instead of
    ``mlp.experts.{e}.{gate,up,down}_proj.weight``. One expert is read as a slice, not the whole tensor.
  * the MXFP8 release (``tencent/Hy4-preview-FP8``, ``quant_algo: MXFP8``) stores fp8 e4m3 weights
    with one E8M0 (uint8, bias 127) scale per 32 elements along the input dim, in ``<key>_scale``
    (``<name>.weight_scale`` / ``experts.gate_up_proj_scale``), instead of DeepSeek's 128 x 128
    ``weight_scale_inv``. The BF16 release has no scales; both go through ``bf16()``.
  * attention: the output gate is ``self_attn.linear_gate`` (cached as ttMLA's ``g_proj``), and each
    layer's learnable per-head sink ``self_attn.learnable_sink_param`` ``[heads]`` is ttMLA's
    ``attn_sink.weight`` (#65921; passed to ``sparse_sdpa`` as ``attention_sink``).
  * iHC (#65891): ``hc_attn_layer.hc_pre.*`` / ``hc_mlp_layer.hc_pre.*`` per layer plus ``model.hc_head.*``, read
    fp32 for the ``ihc`` weight category (``ihc_weights`` per layer, ``ihc_head_weights`` for the head; tt/hy4/ihc.py
    caches them as ``layer_{i}.ihc.pt`` / ``ihc_head.pt``).
  * RoPE frames: Hy4's HF attention ropes its raw projection columns HALF-SPLIT (pairs ``(i, i + d/2)``),
    with no DeepSeek-style interleave reshuffle of ``q_pe`` / ``k_pe``, and its indexer ropes the LAST 64
    of the 128 index-key dims. ttMLA ropes INTERLEAVED pairs ``(2i, 2i + 1)`` and TtIndexer ropes the FIRST
    64 dims, interleaved. ``hy4_mla_weights`` reorders the output rows of the affected projections (and the
    index-key LayerNorm) so the device computes the same attention: see ``mla_rope_to_device_frame`` and
    ``indexer_to_device_frame``. Caches built before that reorder are stale (``HY4_MLA_CACHE_FORMAT``).

Import-light (torch + safetensors only); nothing here opens a device.
"""

from __future__ import annotations

import torch

from models.demos.deepseek_v3_d_p.tt.runners import weight_categories as wc

MXFP8_BLOCK = 32  # elements per E8M0 scale, along the last (input) dim
E8M0_BIAS = 127

# Hy4 self_attn submodule -> ttMLA canonical key (see reference/cpu_deepseek_v32/weights.py CANONICAL_TO_CPU).
# Equal to the checkpoint name everywhere except the output gate (linear_gate -> g_proj) and the indexer
# LayerNorm bias, which ttMLA keeps as a separate tensor.
_MLA_KEYS = {
    "q_a_proj.weight": "q_a_proj.weight",
    "q_a_layernorm.weight": "q_a_layernorm.weight",
    "q_b_proj.weight": "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight": "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight": "kv_a_layernorm.weight",
    "kv_b_proj.weight": "kv_b_proj.weight",
    "o_proj.weight": "o_proj.weight",
}
_INDEXER_KEYS = {
    "indexer.wq_b.weight": "indexer.wq_b.weight",
    "indexer.wk.weight": "indexer.wk.weight",
    "indexer.k_norm.weight": "indexer.k_norm.weight",
    "indexer.k_norm.bias": "indexer.k_norm_bias.weight",
    "indexer.weights_proj.weight": "indexer.weights_proj.weight",
}
_GATE_KEYS = {"linear_gate.weight": "g_proj.weight"}
# Learnable per-head sink: [num_heads] logits, kept fp32 (ttMLA divides by the softmax scale, then bf16) -- #65921.
_SINK_KEYS = {"learnable_sink_param": "attn_sink.weight"}
SINK_KEY = "self_attn.learnable_sink_param"

# iHC tensors (#65891), per layer and for the head.
IHC_LAYER_KEYS = tuple(
    f"{sub}.hc_pre.{field}" for sub in ("hc_attn_layer", "hc_mlp_layer") for field in ("hc_base", "hc_fn", "hc_scale")
)  # #65891
IHC_HEAD_KEYS = ("model.hc_head.hc_head_base", "model.hc_head.hc_head_fn", "model.hc_head.hc_head_scale")


def dequant_mxfp8(weight: torch.Tensor, scale: torch.Tensor, block: int = MXFP8_BLOCK) -> torch.Tensor:
    """bf16 from an MXFP8 tensor: fp8 e4m3 values times 2**(scale - 127), one uint8 E8M0 scale per ``block``
    elements of the last dim. Works on any leading shape (a 2-D weight or a sliced 3-D expert stack)."""
    if scale.dtype != torch.uint8:
        raise TypeError(f"MXFP8 scale must be uint8 (E8M0), got {scale.dtype}")
    if weight.shape[:-1] != scale.shape[:-1] or scale.shape[-1] * block < weight.shape[-1]:
        raise ValueError(
            f"MXFP8 scale {tuple(scale.shape)} does not cover weight {tuple(weight.shape)} at block {block}"
        )
    factor = torch.exp2(scale.to(torch.float32) - E8M0_BIAS)
    factor = factor.repeat_interleave(block, dim=-1)[..., : weight.shape[-1]]
    return (weight.to(torch.float32) * factor).to(torch.bfloat16)


class Hy4Checkpoint(wc._Checkpoint):
    """``_Checkpoint`` over a Hy4 (BF16 or MXFP8) checkpoint that also answers DeepSeek-layout routed-expert
    keys (``model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight``) by slicing the fused tensors."""

    def _scale_key(self, key: str):
        """The MXFP8 scale for ``key``: ``<key>_scale`` (``foo.weight_scale``, ``experts.down_proj_scale``)."""
        cand = f"{key}_scale"
        return cand if cand in self.weight_map else None

    def _get_slice(self, key: str, index) -> torch.Tensor:
        from safetensors import safe_open

        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(f"{key} is not in the checkpoint at {self.path}")
        with safe_open(str(self.path / shard), framework="pt") as f:
            return f.get_slice(key)[index]

    def _read(self, key: str, index=None) -> torch.Tensor:
        """``key`` (optionally sliced along its leading dims) as bf16, dequantizing MXFP8 when a scale exists."""
        t = self.get(key) if index is None else self._get_slice(key, index)
        if t.dtype == torch.float8_e4m3fn:
            scale_key = self._scale_key(key)
            if scale_key is None:
                raise KeyError(f"{key} is fp8 but the checkpoint has no {key}_scale (expected MXFP8)")
            s = self.get(scale_key) if index is None else self._get_slice(scale_key, index)
            return dequant_mxfp8(t, s)
        return t.to(torch.bfloat16)

    @staticmethod
    def _parse_expert_key(key: str):
        # model.layers.{i}.mlp.experts.{e}.{proj}.weight  (also model.mtp_layers.{i}.)
        head, sep, tail = key.partition(".mlp.experts.")
        if not sep:
            return None
        parts = tail.split(".")
        if len(parts) != 3 or not parts[0].isdigit() or parts[2] != "weight" or parts[1] not in wc._PROJ:
            return None
        return head, int(parts[0]), parts[1]

    def routed_expert(self, layer_prefix: str, expert: int, proj: str, intermediate: int) -> torch.Tensor:
        """One routed expert's ``proj`` weight in HF Linear layout (``[out, in]``), from the fused tensors."""
        if proj == "down_proj":
            return self._read(f"{layer_prefix}.mlp.experts.down_proj", (slice(expert, expert + 1),))[0]
        fused = f"{layer_prefix}.mlp.experts.gate_up_proj"
        rows = slice(0, intermediate) if proj == "gate_proj" else slice(intermediate, 2 * intermediate)
        return self._read(fused, (slice(expert, expert + 1), rows))[0]

    def bf16(self, key: str) -> torch.Tensor:
        parsed = self._parse_expert_key(key)
        if parsed is not None and key not in self.weight_map:
            layer_prefix, expert, proj = parsed
            # gate_up_proj stacks gate then up, moe_intermediate rows each; down_proj's last dim is that size.
            intermediate = self._last_dim(f"{layer_prefix}.mlp.experts.down_proj")
            return self.routed_expert(layer_prefix, expert, proj, intermediate)
        return self._read(key)

    def _last_dim(self, key: str) -> int:
        from safetensors import safe_open

        with safe_open(str(self.path / self.weight_map[key]), framework="pt") as f:
            return int(f.get_slice(key).get_shape()[-1])


# ---------------------------------------------------------------------------------------------------------------
# RoPE frames: HF Hy4 (half-split) -> ttMLA / TtIndexer (interleaved)
# ---------------------------------------------------------------------------------------------------------------
# The golden-trace tooling (reference/hy4/golden.py, #65919) defines the same two permutations to read device
# caches back into the HF frame; once both land it should import these instead of keeping its own copy.

# Bumped whenever the MLA / indexer tensors hy4_mla_weights produces change for the same checkpoint. A layer whose
# MLA cache carries no stamp for this value was built by an older reader and is rebuilt (tt/hy4/runtime.py).
#   1: half-split -> interleaved RoPE reorder of q_b_proj / kv_a_proj_with_mqa and the index-key frame change.
HY4_MLA_CACHE_FORMAT = 1


def half_split_to_interleaved_perm(d: int) -> torch.Tensor:
    """``out = x[..., perm]`` turns a half-split RoPE vector (pairs ``(i, i + d/2)``) into an interleaved one
    (pairs ``(2i, 2i + 1)``) at the same frequencies: ``perm = [0, h, 1, h + 1, ...]`` with ``h = d / 2``.
    Same convention as ``utils.test_utils.interleave_pe`` and ``tt.mla.rope.interleaved_perm_matrix``."""
    if d % 2:
        raise ValueError(f"RoPE width must be even, got {d}")
    h = d // 2
    return torch.stack([torch.arange(h), torch.arange(h) + h], dim=-1).reshape(d)


def index_k_device_perm(index_head_dim: int, rope_dim: int) -> torch.Tensor:
    """Index-key column permutation, HF frame -> TtIndexer frame: ``[nope | rope (half-split)]`` ->
    ``[rope (interleaved) | nope]``. Applied to the output rows of ``wk``, to ``k_norm`` weight / bias and to
    each head's rows of ``wq_b``, it makes the device emit (and rope) keys / queries in its own frame."""
    if not 0 < rope_dim <= index_head_dim:
        raise ValueError(f"index rope width {rope_dim} must be in (0, {index_head_dim}]")
    nope = index_head_dim - rope_dim
    return torch.cat([nope + half_split_to_interleaved_perm(rope_dim), torch.arange(nope)])


def _per_head_rows(w: torch.Tensor, head_dim: int, row_perm: torch.Tensor) -> torch.Tensor:
    """Apply ``row_perm`` (over ``head_dim``) to every head's block of output rows of a ``[heads * head_dim, in]``
    Linear weight."""
    if w.shape[0] % head_dim:
        raise ValueError(f"{tuple(w.shape)} rows are not a multiple of the head width {head_dim}")
    heads = w.shape[0] // head_dim
    return w.reshape(heads, head_dim, -1)[:, row_perm].reshape(w.shape).contiguous()


def mla_rope_to_device_frame(weights: dict, *, kv_lora_rank: int, qk_nope_head_dim: int, qk_rope_head_dim: int) -> dict:
    """Reorder the RoPE output rows of ``kv_a_proj_with_mqa`` (``[kv_lora | k_pe]``) and of every head of
    ``q_b_proj`` (``[q_nope | q_pe]``) from half-split to interleaved. q_pe . k_pe is unchanged (both get the same
    permutation and the rotation pairs move with it), and the device's interleaved-roped k_pe equals the HF
    half-split-roped k_pe re-based by ``interleave_pe``, which is how the KVPE cache is compared to the golden."""
    out = dict(weights)
    rp = half_split_to_interleaved_perm(qk_rope_head_dim)
    kv = out["kv_a_proj_with_mqa.weight"]
    if kv.shape[0] != kv_lora_rank + qk_rope_head_dim:
        raise ValueError(f"kv_a_proj_with_mqa {tuple(kv.shape)}: expected {kv_lora_rank} + {qk_rope_head_dim} rows")
    out["kv_a_proj_with_mqa.weight"] = torch.cat([kv[:kv_lora_rank], kv[kv_lora_rank:][rp]]).contiguous()
    if "q_b_proj.weight" in out:
        head = qk_nope_head_dim + qk_rope_head_dim
        row_perm = torch.cat([torch.arange(qk_nope_head_dim), qk_nope_head_dim + rp])
        out["q_b_proj.weight"] = _per_head_rows(out["q_b_proj.weight"], head, row_perm)
    return out


def indexer_to_device_frame(weights: dict, *, index_head_dim: int, rope_dim: int) -> dict:
    """Put the indexer's keys and queries in TtIndexer's frame (``index_k_device_perm``): rows of ``wk`` and of
    every head of ``wq_b``, and the k LayerNorm's weight / bias (LayerNorm commutes with a permutation of its
    features when its affine parameters move with them). q . k per head is unchanged."""
    out = dict(weights)
    perm = index_k_device_perm(index_head_dim, rope_dim)
    wk = out["indexer.wk.weight"]
    if wk.shape[0] != index_head_dim:
        raise ValueError(f"indexer.wk {tuple(wk.shape)}: expected {index_head_dim} rows")
    out["indexer.wk.weight"] = wk[perm].contiguous()
    for key in ("indexer.k_norm.weight", "indexer.k_norm_bias.weight"):
        out[key] = out[key][..., perm].contiguous()
    out["indexer.wq_b.weight"] = _per_head_rows(out["indexer.wq_b.weight"], index_head_dim, perm)
    return out


def _mla_cache_files(cache_path) -> dict[int, list]:
    """``{layer: [tensorbin Path, ...]}`` of every MLA + indexer cache file (``layer_{i}.mla.*``,
    ``layer_{i}.mla_repl.*``, ``layer_{i}.mla_tp.*``; the indexer's files are ``layer_{i}.mla.indexer_*``). One
    directory scan."""
    from pathlib import Path

    files: dict[int, list] = {}
    cache_path = Path(cache_path)
    if not cache_path.is_dir():
        return files
    for f in cache_path.iterdir():
        head, dot, rest = f.name.partition(".")
        if not (dot and head.startswith("layer_") and head[6:].isdigit() and f.name.endswith(".tensorbin")):
            continue
        if rest.split(".", 1)[0] in ("mla", "mla_repl", "mla_tp"):
            files.setdefault(int(head[6:]), []).append(f)
    return files


def mla_cache_stamp(cache_path, layer_idx: int):
    """The marker saying layer ``layer_idx``'s MLA / indexer cache was built at ``HY4_MLA_CACHE_FORMAT``."""
    from pathlib import Path

    return Path(cache_path) / f"layer_{layer_idx}.mla.hy4_format_{HY4_MLA_CACHE_FORMAT}.stamp"


def stale_mla_cache_layers(cache_path, layers) -> list[int]:
    """Layers of ``layers`` that hold MLA / indexer cache files but no stamp for the current format: built by an
    older reader (e.g. before the RoPE frame reorder) and not to be loaded."""
    files = _mla_cache_files(cache_path)
    return [i for i in layers if files.get(i) and not mla_cache_stamp(cache_path, i).exists()]


def purge_mla_cache(cache_path, layers) -> int:
    """Delete the MLA / indexer cache files of ``layers`` so the rebuild writes them afresh (``ttnn.as_tensor``
    loads an existing file instead of converting the weights it is given). Returns the number removed."""
    files = _mla_cache_files(cache_path)
    n = 0
    for i in layers:
        for f in files.get(i, ()):
            f.unlink(missing_ok=True)
            n += 1
        mla_cache_stamp(cache_path, i).unlink(missing_ok=True)
    return n


def stamp_mla_cache(cache_path, layers) -> list[int]:
    """Stamp the layers of ``layers`` whose MLA cache files exist as built at the current format. Returns them."""
    files = _mla_cache_files(cache_path)
    done = [i for i in layers if files.get(i)]
    for i in done:
        mla_cache_stamp(cache_path, i).touch()
    return done


def hy4_mla_weights(
    ckpt: Hy4Checkpoint, layer_prefix: str, *, owns_indexer: bool, gated: bool, config, sink: bool = False
) -> dict:
    """One layer's ttMLA weight dict in canonical names (norms, ``weights_proj`` and the sink fp32, the rest bf16,
    as ``load_attention_state_dict`` produces for GLM), in the device RoPE frames (module docstring). A ``shared``
    indexer layer has no indexer tensors. ``config`` supplies the head dims (the hy_v4 HF config)."""
    names = dict(_MLA_KEYS)
    if owns_indexer:
        names.update(_INDEXER_KEYS)
    if gated:
        names.update(_GATE_KEYS)
    out = {}
    for src, canon in names.items():
        t = ckpt._read(f"{layer_prefix}.self_attn.{src}")
        out[canon] = t.float() if ("norm" in canon or "weights_proj" in canon) else t
    out = mla_rope_to_device_frame(
        out,
        kv_lora_rank=int(config.kv_lora_rank),
        qk_nope_head_dim=int(config.qk_nope_head_dim),
        qk_rope_head_dim=int(config.qk_rope_head_dim),
    )
    if owns_indexer:
        # TtIndexer's rope width (index_rope_dim, default 64); 0 would be a NoPE indexer with nothing to reorder.
        rope_dim = getattr(config, "index_rope_dim", None)
        rope_dim = 64 if rope_dim is None else int(rope_dim)
        if rope_dim:
            if not getattr(config, "index_rope_interleave", True):
                raise ValueError("indexer_to_device_frame targets TtIndexer's interleaved rope (index_rope_interleave)")
            out = indexer_to_device_frame(out, index_head_dim=int(config.index_head_dim), rope_dim=rope_dim)
    if sink:  # stored fp32 and never quantized: read as is, not through _read's bf16 cast
        out[_SINK_KEYS["learnable_sink_param"]] = ckpt.get(f"{layer_prefix}.{SINK_KEY}").float()
    return out


def hy4_layer_state_dict(ckpt: Hy4Checkpoint, config, model_cfg, layer_idx: int, categories: set[str]) -> dict:
    """``weight_categories._layer_state_dict`` for Hy4: the same ``TtPrefillBlock`` state-dict shape, read through
    the Hy4 key layout. Norms, the dense FFN, the router and the shared expert use DeepSeek key names already."""
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import indexer_layer_is_reused

    prefix = f"model.layers.{layer_idx}"
    mla_requested = wc.MLA in categories
    sd = wc._layer_state_dict(ckpt, config, model_cfg, layer_idx, categories - {wc.MLA, wc.ROUTED_EXPERTS, wc.IHC})
    if mla_requested:
        sd["mla_weights"] = hy4_mla_weights(
            ckpt,
            prefix,
            owns_indexer=not indexer_layer_is_reused(config, layer_idx),
            gated=bool(getattr(config, "mla_use_output_gate", False)),
            config=config,
            sink=bool(getattr(config, "mla_learnable_sink", False)),
        )
    if wc.IHC in categories:
        from models.demos.deepseek_v3_d_p.tt.hy4.ihc import read_layer_ihc

        sd["ihc_weights"] = read_layer_ihc(ckpt, layer_idx)
    if wc.ROUTED_EXPERTS in categories:
        intermediate = model_cfg.MOE_INTERMEDIATE_SIZE
        sd["routed_expert_weights"] = [
            {p: ckpt.routed_expert(prefix, e, p, intermediate) for p in wc._PROJ}
            for e in range(model_cfg.NUM_ROUTED_EXPERTS)
        ]
    return sd


def load_hy4_trunk_state_dict(
    checkpoint_path,
    config,
    model_cfg: type,
    *,
    missing: dict[str, list[int]],
    first_layer_idx: int,
    num_layers: int,
) -> dict:
    """``weight_categories.load_trunk_state_dict`` over a Hy4 checkpoint, plus the iHC head when the ``ihc``
    category lists it (``[-1]``)."""
    ckpt = Hy4Checkpoint(checkpoint_path)
    sd = wc.load_trunk_state_dict(
        checkpoint_path,
        config,
        model_cfg,
        missing=missing,
        first_layer_idx=first_layer_idx,
        num_layers=num_layers,
        checkpoint=ckpt,
        layer_state_dict_fn=hy4_layer_state_dict,
    )
    if -1 in missing.get(wc.IHC, ()):
        from models.demos.deepseek_v3_d_p.tt.hy4.ihc import read_head_ihc

        sd["ihc_head_weights"] = read_head_ihc(ckpt)
    return sd
