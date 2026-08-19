# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import os
from typing import Any, NamedTuple

from loguru import logger

import ttnn
from models.common.utility_functions import is_wormhole_b0
from models.demos.qwen3_vl.tt.common import nearest_multiple
from models.tt_transformers.tt.model_config import ModelArgs, OpGroup


class ModelOptimizations:
    def __init__(self, model_name):
        """Configuration optimized for accuracy
        Only 70B models uses bfp4 MLPs in this configuration
        """
        self.bfp4_mlp = False
        # self.bfp4_mlp = "Qwen3-VL-32B" in model_name


class VisionMatmulPlan(NamedTuple):
    """One vision-tower prefill matmul's configuration, from `VisionModelArgs.vision_mm_plan`.

    ``chunk`` is the rows per matmul batch element: the caller reshapes
    ``[1, 1, rows, K] -> [1, rows/chunk, chunk, K]`` first (metadata-only on a TILE tensor) and back
    afterwards. ``program_config`` is None when nothing legal fits, i.e. run on ttnn's auto config.
    """

    chunk: int
    program_config: Any
    compute_kernel_config: Any
    memory_config: Any
    fidelity: str  # which compute_kernel_config_* was chosen, for reports and perf tests
    # Where this matmul wants input 0. A matmul cannot relocate its own input, so the caller hands
    # this to whoever PRODUCES it (the LayerNorm, for qkv and mlp_fc1).
    in0_memory_config: Any = None


_L1_PER_CORE = 1499136  # MEM_L1_SIZE, wormhole/dev_mem_map.h
_L1_RESERVE = 32 * 1024  # the l1_small_size the demo opens with, plus slack

# Per-family matmul tuning, from tests/perf/test_sweep_vision_matmuls.py.
#
# WORMHOLE B0 ONLY -- gated by `self.vision_mm_tuned`. Everything here was swept on N300 and T3K and
# is sized against that hardware: grids and subblocks against an 8x8 core grid, row chunks and L1
# placements against Wormhole's L1 and DRAM bandwidth, the fidelity walk against its throughput.
# Blackhole (P150, P150x4, BH QuietBox) has a 13x10 grid and different DRAM bandwidth, so none of it
# carries over -- there the tower runs exactly as it did before this sweep.
#
# `chunk` and `in0_block_w` are CAPS, snapped down to whatever is legal for the shape actually being
# run, so another image size or TP degree still gets a valid config rather than a crash;
# `in0_block_w=None` leaves that matmul on ttnn's auto config. `in0_l1` and `out_l1` are a TRADE, not
# two free wins: L1 residency for the input, for the output, and the circular buffers all come out of
# the same 1464 KB/core. `grid_x`/`grid_y` pin a grid extent where `_grid_extent`'s divisor rule picks
# the wrong one for a shape.
#
# The measured per-op deltas, the rejected alternatives (sharded activations; in0-in-L1 on the 9B; a
# forced config for the merger) and the headroom deliberately left (LoFi, worth ~-46 ms/tower but an
# accuracy call) are all recorded in ../../VISION_TOWER_PERF.md.
_VISION_MM_TUNING = {
    # patch_embed's DRAM output is deliberate: L1 was worth only -58 us once per image, and its
    # consumers are elementwise ops plus a pad that would inherit L1 unvalidated.
    "patch_embed": dict(in0_l1=False, chunk=5504, in0_block_w=6, fidelity="hifi2", out_l1=False),
    "qkv": dict(in0_l1=False, chunk=1536, in0_block_w=18, fidelity="hifi2", out_l1=False),
    "wo": dict(in0_l1=False, chunk=4096, in0_block_w=24, fidelity="lofi", out_l1=False),
    "mlp_fc1": dict(in0_l1=False, chunk=3072, in0_block_w=6, fidelity="hifi2_fp16", out_l1=False),
    "mlp_fc2": dict(in0_l1=False, chunk=1536, in0_block_w=4, fidelity="hifi2_fp16", out_l1=True),
    # The merger matmuls stay on auto: they already run at ~60% of the FLOP ceiling, and a forced
    # config measured slower in-model on both meshes.
    "merger_fc1": dict(in0_l1=False, chunk=None, in0_block_w=None, fidelity="hifi2_fp16", out_l1=False),
    "merger_fc2": dict(in0_l1=False, chunk=None, in0_block_w=None, fidelity="hifi2_fp16", out_l1=False),
}

# Overrides keyed on `ModelArgs.device_name`, applied on top of the table above, which was swept on
# Qwen3.5-9B / N300 / TP=2. At TP=8 every per-device N is ~4x narrower, which frees the L1 that lets
# qkv and mlp_fc1 hold their INPUT there as well as their output. Taken only where the sweep beat the
# derived config at the same or better fidelity, so each one holds or improves PCC.
_VISION_MM_TUNING_BY_DEVICE = {
    "T3K": {
        "patch_embed": dict(grid_x=8, in0_block_w=6),
        "qkv": dict(chunk=768, grid_x=8, in0_l1=True, out_l1=True),
        "wo": dict(chunk=3072, fidelity="hifi2", out_l1=True),
        "mlp_fc1": dict(chunk=1536, in0_block_w=18, in0_l1=True, out_l1=True),
        # merger_fc1 is NOT overridden: the isolated sweep liked a 2D config (654 -> 555 us) but
        # in-model it measured 559 against auto's 531, and the tower is the number that ships.
        "merger_fc2": dict(chunk=1376, in0_block_w=9, out_l1=True),
    },
}

# qkv and wo are the only families whose PRE-sweep fidelity came from `decoders_optimizations`
# (HiFi4 under this model's preset) rather than from a `compute_kernel_config_*` on the args, so the
# untuned path restores it from there -- preset and all -- rather than hard-coding it. The other five
# families' table entries are already their pre-sweep values, so they need no untuned override.
_UNTUNED_FIDELITY_OP = {"qkv": OpGroup.LI_QKV_PREFILL, "wo": OpGroup.LI_O_PREFILL}
_FIDELITY_NAMES = ("lofi", "hifi2", "hifi2_na", "hifi2_fp16", "hifi2_nol1acc", "hifi4", "hifi4_fp16", "hifi4_fp32")

_TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
    ttnn.float32: 4096,
}


def _divisors(n, hi=None):
    return [d for d in range(1, (n if hi is None else min(n, hi)) + 1) if n % d == 0]


def _grid_extent(tiles, max_extent):
    """Grid extent along one axis.

    A divisor of the tile count wastes no per-core work (a zero-waste 6-wide grid beat a ragged
    8-wide one at N=36 tiles), but a *small* divisor is worse than a ragged full-width grid
    (N=68: 8 beat 4) -- so only take the divisor when it keeps most of the axis.
    """
    divs = _divisors(tiles, max_extent)
    best = max(divs) if divs else 1
    return best if best >= math.ceil(0.75 * max_extent) else max_extent


class VisionModelArgs(ModelArgs):
    # Base __init__ checks the TEXT config's 4 KV heads; the vision tower's own 16 MHA heads
    # (set below) shard exactly at TP=8, so only the base check needs relaxing.
    SUPPORTS_KV_REPLICATION = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The vision tower is always tensor-parallel (Megatron-style): the
        # vision blocks shard their weights across the mesh devices along
        # cluster axis 1.

        # Core dimensions from HF config
        self.dim = self.hf_config.vision_config.hidden_size
        self.unpadded_hidden_dim = self.hf_config.vision_config.intermediate_size
        self.hidden_dim = nearest_multiple(  # pad to a tile multiple per device
            self.unpadded_hidden_dim, self.tile_size * self.num_devices
        )
        if self.hidden_dim != self.unpadded_hidden_dim:
            logger.info(f"padding hidden dim from {self.unpadded_hidden_dim} to {self.hidden_dim}")
        self.head_dim = self.hf_config.vision_config.hidden_size // self.hf_config.vision_config.num_heads
        self.n_heads = self.hf_config.vision_config.num_heads
        self.n_kv_heads = self.hf_config.vision_config.num_heads

        self.padded_head_dim = math.ceil(self.head_dim / self.tile_size) * self.tile_size

        if self.padded_head_dim != self.head_dim:
            logger.info(f"padding head dim from {self.head_dim} to {self.padded_head_dim}")

        self.qkv_size = self.padded_head_dim * (2 * self.n_kv_heads + self.n_heads)
        self.MAX_QKV_MM_SEQ_LEN = self.MAX_QKV_MM_SEQ_LEN

        self.optimizations = ModelOptimizations(
            self.model_name
        )  # todo)) implement finer grained control similar to tt_transformers'

        # The matmul tuning is Wormhole-only (see `_VISION_MM_TUNING`). Off-arch, `vision_mm_plan`
        # returns the untuned plan instead: ttnn's auto config, DRAM in/out, pre-sweep fidelity.
        # `QWEN36_VISION_MM_TUNING=0` forces that path, so it can be exercised on any arch.
        self.vision_mm_tuned = is_wormhole_b0() and os.environ.get("QWEN36_VISION_MM_TUNING", "1") != "0"
        if not self.vision_mm_tuned:
            logger.info(
                f"vision matmul tuning is Wormhole-only; {self.arch_name} keeps the untuned config "
                f"(re-sweep with tests/perf/test_sweep_vision_matmuls.py to tune it)"
            )

        # One plan per (family, rows): the tower rebuilds them 27 times per image otherwise.
        self._vision_mm_plans = {}

        assert self.n_kv_heads % self.cluster_shape[1] == 0, "n_kv_heads must be divisible by num_devices"

        # Sanity-check the divisibility requirements that the TP code relies on.
        tp = self.cluster_shape[1]
        assert self.n_heads % tp == 0, f"vision n_heads ({self.n_heads}) must be divisible by TP={tp}"
        assert self.qkv_size % tp == 0, f"vision qkv_size ({self.qkv_size}) must be divisible by TP={tp}"
        assert self.dim % tp == 0, f"vision dim ({self.dim}) must be divisible by TP={tp}"
        # Can the block I/O contract keep activations FRACTURED along dim=3? Only if dim splits into
        # a whole number of TILES per device: the fracture is restored by tt_all_reduce(dim=3), a
        # reduce_scatter over a TILE-layout tensor, and a tile cannot be split across devices.
        # Unlike hidden_dim (padded to tile_size*num_devices above), dim comes straight from the HF
        # config — Qwen3.6-27B's vision dim 1152 is 36 tiles, i.e. 9/device at TP=4 but 4.5 at TP=8.
        #
        # When it does not divide, run the tower with REPLICATED activations instead: the
        # row-parallel out-projections all-reduce to a full-width replicated tensor rather than
        # reduce-scattering to a fractured one (see vision_ccl.all_reduce_replicated). Weights stay
        # sharded, so no TP compute is given up. The PatchMerger still fractures its OUTPUT for the
        # LLM, which is safe because it splits out_hidden_size (5120 -> 20 tiles/device at TP=8),
        # not dim.
        self.vision_replicated_acts = (self.dim // tp) % self.tile_size != 0
        if self.vision_replicated_acts:
            logger.info(
                f"vision dim {self.dim} is {self.dim / self.tile_size:g} tiles, which TP={tp} cannot "
                f"split into whole tiles ({self.dim / tp / self.tile_size:g}/device) — keeping vision "
                f"activations replicated (weights stay sharded)"
            )
        assert self.hidden_dim % tp == 0, f"vision hidden_dim ({self.hidden_dim}) must be divisible by TP={tp}"
        # PatchMerger shards the merger MLP Megatron-style; its post-shuffle
        # inner dim (mlp_size = hidden * spatial_merge_size^2) and the final
        # out_hidden_size must both divide cleanly.
        vision_cfg = self.hf_config.vision_config
        mlp_size = vision_cfg.hidden_size * (vision_cfg.spatial_merge_size**2)
        out_hidden_size = vision_cfg.out_hidden_size
        assert mlp_size % tp == 0, f"vision merger mlp_size ({mlp_size}) must be divisible by TP={tp}"
        assert out_hidden_size % tp == 0, f"vision out_hidden_size ({out_hidden_size}) must be divisible by TP={tp}"

    # ------------------------------------------------------------------ prefill matmul planning

    def vision_mm_plan(
        self,
        family: str,
        *,
        rows: int,
        k: int,
        n: int,
        in0_dtype,
        in1_dtype,
        out_dtype,
        fused_activation=None,
        in0_already_l1: bool = False,
    ) -> VisionMatmulPlan:
        """2D-mcast plan for one vision-tower matmul, sized from its ACTUAL per-device shape.

        Args:
            family: key into ``_VISION_MM_TUNING`` (``qkv``, ``wo``, ``mlp_fc1``, ...).
            rows: total activation rows (the padded sequence length; the merged patch count for the
                merger). k, n: per-device contraction and output widths.
            in0_dtype, in1_dtype, out_dtype: needed to size the circular buffers.
            fused_activation: ``ttnn.UnaryWithParam`` to fold in. Pass it HERE, not as
                ``ttnn.linear(activation=...)``: with no explicit core grid that kwarg runs as a
                separate ``unary_chain`` op (matmul.cpp) -- 1.2 ms/block for the MLP's GELU.
            in0_already_l1: set when the PRODUCER already writes in0 into L1 (mlp_fc2 reads mlp_fc1's
                L1 output). in0 residency is not this matmul's choice then, but it still spends the L1
                the output placement is budgeted against.

        Everything is checked against the L1 budget; if nothing legal fits the plan falls back to
        ttnn's auto config and an unchunked activation, which is what the tower did before tuning.
        Off Wormhole (`vision_mm_tuned=False`) that fallback is all this returns.
        """
        cache_key = (family, rows, k, n, in0_dtype, in1_dtype, out_dtype, repr(fused_activation), in0_already_l1)
        cached = self._vision_mm_plans.get(cache_key)
        if cached is not None:
            return cached

        tune = dict(_VISION_MM_TUNING[family])
        if self.vision_mm_tuned:
            tune.update(_VISION_MM_TUNING_BY_DEVICE.get(self.device_name, {}).get(family, {}))
            untuned_op = None
        else:
            untuned_op = _UNTUNED_FIDELITY_OP.get(family)
        if untuned_op is not None:
            # Preset-resolved, so name it by identity rather than re-deriving which setting it is.
            ckc = self.decoders_optimizations.get_math_fidelity(decoder_id=0, op=untuned_op, configuration=self)
            fidelity = next((f for f in _FIDELITY_NAMES if getattr(self, f"compute_kernel_config_{f}") is ckc), "hifi4")
        else:
            ckc = getattr(self, f"compute_kernel_config_{tune['fidelity']}")
            fidelity = tune["fidelity"]
        dram = ttnn.DRAM_MEMORY_CONFIG
        auto = VisionMatmulPlan(
            chunk=rows,
            program_config=None,
            compute_kernel_config=ckc,
            memory_config=dram,
            fidelity=fidelity,
            in0_memory_config=dram,
        )
        self._vision_mm_plans[cache_key] = auto  # replaced below if a 2D config is legal
        if not self.vision_mm_tuned or tune["in0_block_w"] is None:
            return auto

        tile = self.tile_size
        assert rows % tile == 0 and k % tile == 0 and n % tile == 0, f"{family}: {rows}x{k}x{n} not tile-aligned"
        k_t, n_t = k // tile, n // tile
        grid = self.mesh_device.compute_with_storage_grid_size()

        # Largest legal chunk at or below the swept cap (`None` == do not chunk at all). Must divide
        # `rows` so the reshape is exact.
        chunk_cap = tune["chunk"] or rows
        chunk = max(c * tile for c in _divisors(rows // tile, max(1, chunk_cap // tile)))
        m_t = chunk // tile

        gx = min(tune.get("grid_x") or _grid_extent(n_t, grid.x), grid.x)
        gy = min(tune.get("grid_y") or _grid_extent(m_t, grid.y), grid.y)
        per_core_m, per_core_n = math.ceil(m_t / gy), math.ceil(n_t / gx)
        cap = 4 if ckc.fp32_dest_acc_en else 8  # DST capacity; fp32 accumulate halves it

        # Widest legal `out_subblock_w` first -- that, not the largest h*w area, won at every family.
        subblocks = sorted(
            ((h, w) for h in _divisors(per_core_m) for w in _divisors(per_core_n) if h * w <= cap),
            key=lambda hw: (-hw[1], -hw[0] * hw[1]),
        )
        if not subblocks:
            return auto

        def cb_bytes(in0_block_w):
            in0 = per_core_m * in0_block_w * _TILE_BYTES[in0_dtype] * 2
            in1 = in0_block_w * per_core_n * _TILE_BYTES[in1_dtype] * 2
            out = per_core_m * per_core_n * _TILE_BYTES[out_dtype]
            # The intermediate CB ALIASES the output CB when their formats match (bfloat16 out, no
            # fp32 accumulate). Counting it twice cost the 27B's mlp_fc2 its L1 output, 298 -> 493 us.
            if ckc.fp32_dest_acc_en:
                interm = per_core_m * per_core_n * 4096
            elif out_dtype is ttnn.bfloat16:
                interm = 0
            else:
                interm = per_core_m * per_core_n * _TILE_BYTES[out_dtype]
            return in0 + in1 + out + interm

        # in0_block_w must divide K_tiles; take the largest at or below the cap that fits L1. A prime
        # K_tiles (27B/TP=8 has one) has no divisor below the cap but 1 -- take the whole K instead.
        candidates = _divisors(k_t, tune["in0_block_w"])
        if candidates == [1] and k_t > 1:
            candidates = [k_t]
        in0_block_w = None
        for cand in sorted(candidates, reverse=True):
            if cb_bytes(cand) <= _L1_PER_CORE - _L1_RESERVE:
                in0_block_w = cand
                break
        if in0_block_w is None:
            logger.info(f"vision {family}: {rows}x{k}x{n} has no L1-legal 2D config, leaving it on auto")
            return auto

        # in0 and the output share whatever the CBs leave; claim in0 first (the table only asks for it
        # where it beat the output), then give the output the rest. An L1-interleaved buffer is paged
        # across the grid, so its per-core cost is total/num_cores.
        free_l1 = (_L1_PER_CORE - _L1_RESERVE - cb_bytes(in0_block_w)) * grid.x * grid.y
        in0_bytes = rows * k * _TILE_BYTES[in0_dtype] // (tile * tile)
        in0_cfg = dram
        if in0_already_l1:
            in0_cfg = ttnn.L1_MEMORY_CONFIG  # not our choice; the producer put it there
            free_l1 -= in0_bytes
        elif tune["in0_l1"] and in0_bytes <= free_l1:
            in0_cfg = ttnn.L1_MEMORY_CONFIG
            free_l1 -= in0_bytes
        out_bytes = rows * n * _TILE_BYTES[out_dtype] // (tile * tile)
        mem_cfg = ttnn.L1_MEMORY_CONFIG if (tune["out_l1"] and 0 <= free_l1 and out_bytes <= free_l1) else dram

        sbh, sbw = subblocks[0]
        plan = VisionMatmulPlan(
            chunk=chunk,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=in0_block_w,
                out_subblock_h=sbh,
                out_subblock_w=sbw,
                per_core_M=per_core_m,
                per_core_N=per_core_n,
                transpose_mcast=False,
                fused_activation=fused_activation,
                fuse_batch=rows == chunk,
            ),
            compute_kernel_config=ckc,
            memory_config=mem_cfg,
            fidelity=tune["fidelity"],
            in0_memory_config=in0_cfg,
        )
        logger.info(
            f"vision {family}: {rows}x{k}x{n} -> chunk {chunk}, grid {gx}x{gy}, "
            f"in0_block_w {in0_block_w}, subblock {sbh}x{sbw}, per_core {per_core_m}x{per_core_n}, "
            f"{tune['fidelity']}, in0 {'L1' if in0_cfg is ttnn.L1_MEMORY_CONFIG else 'DRAM'}, "
            f"out {'L1' if mem_cfg is ttnn.L1_MEMORY_CONFIG else 'DRAM'}"
            f"{f' [{self.device_name} override]' if family in _VISION_MM_TUNING_BY_DEVICE.get(self.device_name, {}) else ''}"
        )
        self._vision_mm_plans[cache_key] = plan
        return plan

    def prepare_residual_tensor_prefill(self, x_bsh):
        """
        Prepare inputs for prefill mode.
        x: (batch, seq, hidden_dim)
        B: batch (1)
        S: sequence len
        H: dim

        The vision blocks consume tensors fractured along the hidden dim
        (dim=3 of the 4D tensor), so we shard at load time across cluster
        axis 1 — unless the tower runs with replicated activations
        (``vision_replicated_acts``, when TP cannot split dim into whole
        tiles), in which case every device gets the full hidden dim.
        """

        x_1BSH = x_bsh.unsqueeze(0)

        mesh_mapper = (
            ttnn.ReplicateTensorToMesh(self.mesh_device)
            if self.vision_replicated_acts
            else ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(None, -1),
                mesh_shape=self.cluster_shape,
            )
        )

        # input goes to DRAM
        xs_1BSH = ttnn.from_torch(
            x_1BSH,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        return xs_1BSH

    # Visual model does not use distributed norm for now
    def is_distributed_norm(self, mode):
        return False

    def get_state_dict_prefix(self, module_name, layer_num=None, deepstack_merger_num=None):
        layer_prefix = f"visual.blocks.{layer_num}." if layer_num is not None else ""
        module_map = {
            "MLP": "feed_forward",
            "VisionAttention": "attention",
            "VisionBlock": "",
            "VisionTransformer": "visual",
            "PatchMerger": "visual.merger",
            "norm1": "norm1",
            "norm2": "norm2",
            "DeepstackMerger": f"visual.deepstack_merger_list.{deepstack_merger_num}",
            "": "",  # If no module is given, just get layer prefix
        }
        return layer_prefix + module_map[module_name]

    def reference_vision_model(self, depth=None):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as AutoModelForCausalLM

        print("Loading Qwen3.5 model: ", AutoModelForCausalLM)
        config = AutoModelForCausalLM.config_class.from_pretrained(self.CKPT_DIR)
        config.vision_config.depth = depth if depth is not None else config.vision_config.depth
        model = AutoModelForCausalLM.from_pretrained(self.CKPT_DIR, config=config)
        return model.model.visual

    def reference_vision_block(self, layer_num=0):
        return self.reference_vision_model().blocks[layer_num]

    def reference_mlp(self):
        return self.reference_vision_block().mlp

    def reference_attention(self):
        return self.reference_vision_block().attn

    def reference_rms_norm(self):
        return self.reference_vision_block().norm2

    def reference_patch_merger(self):
        return self.reference_vision_model().merger

    def reference_patch_embed(self):
        return self.reference_vision_model().patch_embed
