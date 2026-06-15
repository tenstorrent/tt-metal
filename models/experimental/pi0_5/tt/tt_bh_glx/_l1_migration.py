# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Post-construction DRAM->L1 migration for the denoise stage.

The denoise loop runs N (5-10) Euler steps; on every step each of the 6
denoise chips re-reads its ~93 MB of expert weights. Those weights are
uploaded to DRAM by default (see expert_slice/suffix_slice/pipeline), so the
per-step matmuls stream them from DRAM each iteration. Moving the static
denoise weights into L1 once at pipeline construction lets the per-step
matmuls read them from on-chip L1 instead.

This fits: 3 expert layers/chip (~93 MB bf8 matmul + bf16 mod) sit inside the
~180 MB usable L1 per Blackhole chip with headroom for the (small) expert
matmul CB regions. The expert MLP (4096) is 4x smaller than the VLM MLP
(16384), so unlike the prefill stage the interleaved-L1 weights don't clash
with the kernel's static CB region.

Gated by PI0_GLX_DENOISE_L1 (default ON). Set =0 to keep weights in DRAM.
"""

from __future__ import annotations

import os
from typing import Optional

import ttnn


def denoise_l1_enabled() -> bool:
    """Whether to place denoise-stage weights in L1. Default ON."""
    return os.environ.get("PI0_GLX_DENOISE_L1", "1").lower() in ("1", "true", "yes", "on")


def siglip_l1_enabled() -> bool:
    """Whether to place GLX SigLIP matmul weights in L1. Default OFF."""
    return os.environ.get("PI0_GLX_SIGLIP_L1", "0").lower() in ("1", "true", "yes", "on")


def prefill_vlm_l1_enabled() -> bool:
    """Whether to place GLX VLM block matmul weights in L1. Default OFF."""
    return os.environ.get("PI0_GLX_PREFILL_VLM_L1", "0").lower() in ("1", "true", "yes", "on")


def prefill_vlm_l1_projs() -> tuple:
    """Which VLM matmul projections to place in L1.

    Env PI0_GLX_PREFILL_VLM_L1_PROJ is a comma list. Defaults to every VLM
    block matmul weight: fused attention QKV, attention output projection, and
    MLP gate/up/down.
    """
    raw = os.environ.get("PI0_GLX_PREFILL_VLM_L1_PROJ", "wqkv,o_proj,gate_proj,up_proj,down_proj")
    return tuple(p.strip() for p in raw.split(",") if p.strip())

def siglip_mlp_l1_enabled() -> bool:
    """Whether to place SigLIP MLP (fc1+fc2) weights in L1. Default OFF.

    ⚠ INTERLEAVED-L1 DOES NOT WORK FOR SIGLIP (verified 2026-06-13). Per-chip
    capacity fits (9 × 10.8 MB ≈ 97 MB of 168 MB usable), but enabling this
    flag trips the matmul kernel's static-CB-vs-interleaved-L1-buffer clash:
        TT_THROW: Statically allocated circular buffers in program N clash
        with L1 buffers on core range [0-0 - 11-1]. L1 buffer allocated at
        591616 and static circular buffer region ends at 733696
    SigLIP intermediate (4608) crosses the same threshold the existing comment
    at the top of this file flags for VLM (16384) — the matmul kernel's CB
    footprint stops fitting alongside interleaved L1 weights on the same cores.

    To actually land SigLIP weights on L1 you need WIDTH-SHARDED layout +
    a matmul kernel that consumes width-sharded weights (same blocker
    PI0_GLX_PREFILL_MLP_L1 has). The flag + migrator stay in place as an
    opt-in for future width-sharded experiments — do not turn on with the
    interleaved migrator below."""
    return os.environ.get("PI0_GLX_SIGLIP_MLP_L1", "0").lower() in ("1", "true", "yes", "on")


def siglip_attn_l1_enabled() -> bool:
    """Whether to ALSO place SigLIP attention (wqkv + wo) weights in L1.

    Default OFF. Stacking attn on top of MLP brings per-chip footprint to
    ~163 MB ≈ 97% of usable L1 — leaves no headroom for matmul CBs +
    activations. Only enable if you've verified L1 isn't exhausted on the
    target chip layout."""
    return os.environ.get("PI0_GLX_SIGLIP_ATTN_L1", "0").lower() in ("1", "true", "yes", "on")


def prefill_mlp_l1_enabled() -> bool:
    """Whether to width-shard the prefill VLM MLP weights into L1. Default OFF.

    Width-sharded weights require a matmul that consumes them; the current
    MinimalMatmul / ttnn.linear path expects interleaved weights, so this is
    opt-in until the faster width-sharded-aware matmul is wired."""
    return os.environ.get("PI0_GLX_PREFILL_MLP_L1", "0").lower() in ("1", "true", "yes", "on")


def prefill_mlp_l1_projs() -> tuple:
    """Which MLP projections to place in L1. Env PI0_GLX_PREFILL_MLP_L1_PROJ
    (comma list).

    Verified safe (2026-06-13, layout='interleaved'):
      - down_proj alone:        works (N=2048, small CB despite K=16384)
    Verified crashes (N=16384 → static CB region too large):
      - up_proj or gate_proj    (any combination including either)

    Default is 'down_proj' only — the largest weight that fits with the normal
    matmul kernel. Use 'gate_proj,up_proj,down_proj' for width-shard layout."""
    raw = os.environ.get("PI0_GLX_PREFILL_MLP_L1_PROJ", "down_proj")
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def prefill_mlp_l1_layout() -> str:
    """'interleaved' (for the normal matmul) or 'width_sharded'. Env
    PI0_GLX_PREFILL_MLP_L1_LAYOUT. Default 'interleaved' — the normal
    MatmulMultiCoreReuseMultiCast / ttnn.linear path reads interleaved weights."""
    return os.environ.get("PI0_GLX_PREFILL_MLP_L1_LAYOUT", "interleaved").strip().lower()


def prefill_mlp_l1_grid() -> tuple:
    """(grid_x, grid_y) for prefill MLP width-sharding. Env PI0_GLX_PREFILL_MLP_L1_GRID='gx,gy'.

    Default 12x10=120 (the full BH Tensix grid the matmul reports). N=16384 is
    512 tiles, which is NOT divisible by 120, so the shard is uneven/padded:
    ceil(512/120)=5 tiles/core → padded N=19200 (~17% waste). The padded shard
    fits L1 (~1.25 MB/core for gate+up+down at bf8). For an even (un-padded)
    shard use a divisor-clean grid (8x8=64), but that overflows L1 with all
    three weights — see the budget guard in migrate_prefill_mlp_weights_to_l1."""
    raw = os.environ.get("PI0_GLX_PREFILL_MLP_L1_GRID", "12,10")
    gx, gy = (int(x) for x in raw.split(","))
    return gx, gy


def _to_l1(t: Optional["ttnn.Tensor"]) -> Optional["ttnn.Tensor"]:
    """Move a tensor to L1 and free the DRAM source; idempotent on L1 tensors.

    When the tensor is already L1-resident, to_memory_config returns the same
    buffer reference, so deallocating it would dangle the returned handle.
    Guard against that by checking buffer_type first.
    """
    if t is None:
        return None
    if t.memory_config().buffer_type == ttnn.BufferType.L1:
        return t
    moved = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(t)
    return moved


def _migrate_expert_block(block) -> None:
    """Move one AdaRMSGemmaBlockTTNN's weights+biases to L1."""
    attn = getattr(block, "attention", None)
    if attn is not None:
        attn.wqkv = _to_l1(attn.wqkv)
        attn.o_proj = _to_l1(attn.o_proj)
    mlp = getattr(block, "mlp", None)
    if mlp is not None:
        mlp.gate_proj = _to_l1(mlp.gate_proj)
        mlp.up_proj = _to_l1(mlp.up_proj)
        mlp.down_proj = _to_l1(mlp.down_proj)
    block.mod_weight = _to_l1(block.mod_weight)
    block.mod_bias = _to_l1(block.mod_bias)


def _migrate_gemma_block_matmuls(block, projs: tuple = ("wqkv", "o_proj", "gate_proj", "up_proj", "down_proj")) -> None:
    """Move GemmaBlockTTNN matmul weights selected by projection name to L1."""
    attn = getattr(block, "attention", None)
    if attn is not None:
        if "wqkv" in projs:
            attn.wqkv = _to_l1(attn.wqkv)
        if "o_proj" in projs:
            attn.o_proj = _to_l1(attn.o_proj)
    mlp = getattr(block, "mlp", None)
    if mlp is not None:
        for name in ("gate_proj", "up_proj", "down_proj"):
            if name in projs:
                setattr(mlp, name, _to_l1(getattr(mlp, name)))


def _migrate_siglip_block_matmuls(block) -> None:
    """Move SigLIPBlockTTNN matmul weights and their biases to L1."""
    attn = getattr(block, "attention", None)
    if attn is not None:
        attn.wqkv = _to_l1(attn.wqkv)
        attn.bqkv = _to_l1(getattr(attn, "bqkv", None))
        attn.wo = _to_l1(attn.wo)
        attn.bo = _to_l1(getattr(attn, "bo", None))
    mlp = getattr(block, "mlp", None)
    if mlp is not None:
        for name in ("fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"):
            setattr(mlp, name, _to_l1(getattr(mlp, name, None)))


def migrate_siglip_weights_to_l1(stage_vision) -> None:
    """Move GLX SigLIP matmul weights owned by vision slices to L1.

    Covers patch embedding, every SigLIP block QKV/O/MLP matmul weight, and the
    multimodal projector. Position embeddings and layernorm tensors are not
    matmul weights and are intentionally left unchanged.
    """
    embed = getattr(stage_vision, "embed_slice", None)
    patch = getattr(embed, "patch_embed", None)
    if patch is not None:
        patch._linear_weight = _to_l1(getattr(patch, "_linear_weight", None))
        patch._linear_bias = _to_l1(getattr(patch, "_linear_bias", None))
        if getattr(patch, "_use_fold", False):
            patch._fold_weight = _to_l1(getattr(patch, "_fold_weight", None))
            patch._fold_bias = _to_l1(getattr(patch, "_fold_bias", None))

    for slice_name in ("layer_slice_a", "layer_slice_b", "tail_slice"):
        sl = getattr(stage_vision, slice_name, None)
        for block in getattr(sl, "blocks", []):
            _migrate_siglip_block_matmuls(block)

    tail = getattr(stage_vision, "tail_slice", None)
    projector = getattr(tail, "mm_projector", None)
    if projector is not None:
        projector.weight = _to_l1(getattr(projector, "weight", None))
        projector.bias = _to_l1(getattr(projector, "bias", None))


def migrate_prefill_vlm_weights_to_l1(stage_prefill, projs: tuple) -> None:
    """Move selected GLX VLM block matmul weights into interleaved L1."""
    for sl in getattr(stage_prefill, "slices", []):
        _migrate_gemma_block_matmuls(sl.block, projs)


def migrate_denoise_weights_to_l1(stage_denoise, suffix_slices, denoise_head) -> None:
    """Move every static denoise-stage weight/bias from DRAM to L1.

    Covers, per the denoise stage's owned tensors:
      - each expert chunk's blocks (QKV/O, MLP gate/up/down, adaRMS mod w+b)
      - each expert chunk's RoPE cos/sin tables (re-read every step)
      - each replicated suffix MLP's weights + biases
      - the final adaRMS-norm dense weight + bias (last denoise chip)

    Activations stay where the forward path already puts them (L1 in the
    denoise loop + suffix matmul outputs); only the DRAM-resident static
    tensors are migrated here.
    """
    for chunk in stage_denoise.chunks:
        for block in chunk.blocks:
            _migrate_expert_block(block)
        chunk.cos_meta = _to_l1(chunk.cos_meta)
        chunk.sin_meta = _to_l1(chunk.sin_meta)
        # The chunk's 3 blocks captured the pre-migration cos/sin tensors at
        # construction (GemmaAttentionTTNN stores its own self.cos_meta ref).
        # _to_l1 freed those DRAM sources, so repoint every block's attention
        # at the migrated L1 tensors — otherwise the per-step RoPE slice in the
        # no-override path (ttnn_gemma.py:700) reads a deallocated buffer and
        # raises "Tensor is not allocated".
        for block in chunk.blocks:
            block.attention.cos_meta = chunk.cos_meta
            block.attention.sin_meta = chunk.sin_meta

    for sl in suffix_slices:
        weights = sl.suffix.weights
        for key in list(weights.keys()):
            weights[key] = _to_l1(weights[key])

    denoise_head.mod_weight = _to_l1(denoise_head.mod_weight)
    denoise_head.mod_bias = _to_l1(denoise_head.mod_bias)


_SIGLIP_VALID = {"fc1", "fc2", "wqkv", "wo"}
_VLM_ATTN_VALID = {"wqkv", "o_proj"}


def migrate_siglip_weights_to_l1(stage_vision, tensors=None, include_attn=None) -> None:
    """Move a subset of SigLIP weights (per-tensor allowlist) from DRAM to L1.

    tensors: iterable from {"fc1", "fc2", "wqkv", "wo"}. None → no-op.
    include_attn: legacy bool; True → adds {"wqkv", "wo"} to the set.

    Verified (2026-06-13, e2e perf test):
      - wo + fc2 (N=1152):     PASS (small kernel CB, weights fit alongside)
      - fc1 alone (N=4608):    CRASH ("Statically allocated CBs ... clash with
                               L1 buffers ... CB region ends at 464384")
      - wqkv alone (N=4608):   CRASH (same threshold as fc1)

    The safe-stack default in pipeline.py is wo+fc2. fc1/wqkv require a
    width-shard-aware matmul to ever land on L1.
    """
    if include_attn:
        tensors = set(tensors or []) | {"wqkv", "wo"}
    tensors = set(tensors or [])
    bad = tensors - _SIGLIP_VALID
    if bad:
        raise ValueError(f"unknown SigLIP tensors {bad}; valid: {_SIGLIP_VALID}")
    if not tensors:
        return
    for name in ("layer_slice_a", "layer_slice_b", "tail_slice"):
        sl = getattr(stage_vision, name, None)
        if sl is None or not hasattr(sl, "blocks"):
            continue
        for block in sl.blocks:
            mlp = getattr(block, "mlp", None)
            attn = getattr(block, "attention", None)
            if "fc1" in tensors and mlp is not None:
                mlp.fc1_weight = _to_l1(mlp.fc1_weight)
                mlp.fc1_bias = _to_l1(mlp.fc1_bias)
            if "fc2" in tensors and mlp is not None:
                mlp.fc2_weight = _to_l1(mlp.fc2_weight)
                mlp.fc2_bias = _to_l1(mlp.fc2_bias)
            if "wqkv" in tensors and attn is not None:
                attn.wqkv = _to_l1(attn.wqkv)
                attn.bqkv = _to_l1(getattr(attn, "bqkv", None))
            if "wo" in tensors and attn is not None:
                attn.wo = _to_l1(attn.wo)
                attn.bo = _to_l1(getattr(attn, "bo", None))


def vlm_attn_l1_enabled() -> bool:
    """Move VLM PaliGemma attention weights (wqkv + o_proj) to L1. Default OFF.

    Per-chip footprint (1 layer/chip across 18 chips): wqkv 5.4 MB + o_proj
    4.3 MB = 9.7 MB. Tiny vs 168 MB usable L1. Verified PASS (2026-06-13) —
    N=2560 wqkv and N=2048 o_proj kernel CBs fit alongside L1 weights."""
    return os.environ.get("PI0_GLX_VLM_ATTN_L1", "0").lower() in ("1", "true", "yes", "on")


def vlm_attn_l1_tensors() -> tuple:
    """Which VLM attn weights to migrate. Env PI0_GLX_VLM_ATTN_L1_TENSORS=wqkv,o_proj.
    Default both."""
    raw = os.environ.get("PI0_GLX_VLM_ATTN_L1_TENSORS", "wqkv,o_proj")
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def siglip_l1_tensors() -> tuple:
    """SigLIP per-tensor allowlist. Env PI0_GLX_SIGLIP_L1_TENSORS=wo,fc2 (or
    any subset of {wo, fc2, wqkv, fc1}). Empty/unset → no migration."""
    raw = os.environ.get("PI0_GLX_SIGLIP_L1_TENSORS", "")
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def migrate_vlm_attn_to_l1(stage_prefill, tensors=("wqkv", "o_proj")) -> None:
    """Move VLM attention weights (wqkv, o_proj) from DRAM to L1.

    Each prefill chip holds one VLMBlockSlice whose .block.attention is a
    GemmaAttentionTTNN with attributes wqkv (1[1]×N=2560) and o_proj (N=2048).
    Both are small-N — the matmul kernel CB region fits alongside interleaved
    L1 weights on the same cores (unlike the gate/up/down MLPs at N=16384)."""
    bad = set(tensors) - _VLM_ATTN_VALID
    if bad:
        raise ValueError(f"unknown VLM tensors {bad}; valid: {_VLM_ATTN_VALID}")
    if not tensors:
        return
    for slc in getattr(stage_prefill, "slices", []) or []:
        block = getattr(slc, "block", None)
        attn = getattr(block, "attention", None) if block else None
        if attn is None:
            continue
        if "wqkv" in tensors:
            attn.wqkv = _to_l1(attn.wqkv)
        if "o_proj" in tensors:
            attn.o_proj = _to_l1(attn.o_proj)


# ---------------------------------------------------------------------------- #
# Prefill VLM MLP — width-sharded L1 (opt-in, for a width-shard-aware matmul)   #
# ---------------------------------------------------------------------------- #

# bf8_b effective bytes/element: 1 data byte + 1 exponent byte per 16-element
# block ≈ 17/16. Used only for the per-core L1 budget guard.
_BF8_BYTES_PER_ELEM = 17.0 / 16.0
# Conservative usable L1 per Blackhole Tensix core (1.5 MB minus CB/runtime).
_USABLE_L1_PER_CORE = 1_400_000


def _shard_width_tiles(n: int, num_cores: int) -> int:
    """Tiles of width each core holds (ceil) — supports uneven/padded sharding."""
    n_tiles = n // 32
    return -(-n_tiles // num_cores)  # ceil division


def _to_l1_width_sharded(t, grid_x: int, grid_y: int):
    """Move a 2D weight to WIDTH_SHARDED L1 on a grid_x*grid_y core grid.

    Each core owns a [K, shard_w] column slice where shard_w = ceil(N_tiles /
    num_cores) * 32. When N_tiles isn't divisible by num_cores the shard is
    padded (num_cores*shard_w > N) — the trailing cores hold padding. Frees the
    DRAM source. Idempotent on tensors already in L1.
    """
    if t is None:
        return None
    if t.memory_config().buffer_type == ttnn.BufferType.L1:
        return t
    shape = list(t.shape)
    while len(shape) > 2 and shape[0] == 1:
        shape = shape[1:]
    if len(shape) != 2:
        raise ValueError(f"_to_l1_width_sharded expects a 2D weight, got {list(t.shape)}")
    k, n = shape
    if n % 32 != 0:
        raise ValueError(f"_to_l1_width_sharded: N={n} not tile-aligned (mod 32)")
    num_cores = grid_x * grid_y
    shard_w = _shard_width_tiles(n, num_cores) * 32
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    shard_spec = ttnn.ShardSpec(grid, (k, shard_w), ttnn.ShardOrientation.ROW_MAJOR)
    memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    moved = ttnn.to_memory_config(t, memcfg)
    ttnn.deallocate(t)
    return moved


# Aggregate usable L1 per BH chip: 120-core compute grid * 1.5 MB, derated to
# ~1.4 MB/core for CB/runtime reservations.
_AGG_L1_PER_CHIP = 120 * _USABLE_L1_PER_CORE


def migrate_prefill_mlp_weights_to_l1(stage_prefill, layout: str, grid_x: int, grid_y: int, projs: tuple) -> None:
    """Move the selected prefill VLM block MLP weights into L1.

    `projs` is the subset of ('gate_proj','up_proj','down_proj') to migrate;
    the rest stay in DRAM. Gemma MLP has no biases. layout='interleaved'
    (for the normal matmul) bank-interleaves; 'width_sharded' gives each core a
    [K, N/cores] slice (for a width-shard-aware matmul). Both raise (not
    silently OOM) if the migrated weights overflow the relevant L1 budget.
    """
    if not stage_prefill.slices:
        return
    mlp0 = stage_prefill.slices[0].block.mlp
    sel0 = [getattr(mlp0, p) for p in projs]

    if layout == "width_sharded":
        num_cores = grid_x * grid_y
        per_core = 0.0
        for w in sel0:
            k, n = [d for d in list(w.shape) if d != 1][-2:]
            per_core += (k * _shard_width_tiles(n, num_cores) * 32) * _BF8_BYTES_PER_ELEM
        if per_core > _USABLE_L1_PER_CORE:
            raise RuntimeError(
                f"prefill MLP width-shard ({','.join(projs)}) needs ~{per_core/1e6:.2f} MB/core on "
                f"{grid_x}x{grid_y}={num_cores} cores, over ~{_USABLE_L1_PER_CORE/1e6:.2f} MB/core. "
                f"Use a larger grid (PI0_GLX_PREFILL_MLP_L1_GRID) or fewer projs."
            )
        for sl in stage_prefill.slices:
            mlp = sl.block.mlp
            for p in projs:
                setattr(mlp, p, _to_l1_width_sharded(getattr(mlp, p), grid_x, grid_y))
    else:  # interleaved
        agg = sum((k := [d for d in list(w.shape) if d != 1])[-2] * k[-1] * _BF8_BYTES_PER_ELEM for w in sel0)
        if agg > _AGG_L1_PER_CHIP:
            raise RuntimeError(
                f"prefill MLP interleaved-L1 ({','.join(projs)}) needs ~{agg/1e6:.0f} MB/chip, over the "
                f"~{_AGG_L1_PER_CHIP/1e6:.0f} MB aggregate L1. Note: even when the weights fit, the normal "
                f"matmul's static CB region must also fit per-core — see the gate+up-only note."
            )
        for sl in stage_prefill.slices:
            mlp = sl.block.mlp
            for p in projs:
                setattr(mlp, p, _to_l1(getattr(mlp, p)))
