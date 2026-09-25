# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import os
from typing import Any, NamedTuple

from loguru import logger

import ttnn
from models.common.utility_functions import is_wormhole_b0
from models.demos.qwen3_vl.tt.common import nearest_multiple
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import ModelArgs, OpGroup

from .vision_ccl import vision_ccl_kwargs as _vision_ccl_kwargs


class ModelOptimizations:
    def __init__(self, model_name):
        """Configuration optimized for accuracy
        Only 70B models uses bfp4 MLPs in this configuration
        """
        self.bfp4_mlp = False
        # self.bfp4_mlp = "Qwen3-VL-32B" in model_name


class VisionMatmulPlan(NamedTuple):
    """One vision-tower prefill matmul plan; program_config is None on the auto fallback."""

    chunk: int
    program_config: Any
    compute_kernel_config: Any
    memory_config: Any
    fidelity: str
    # Matmul cannot move input 0; the producer writes in0_memory_config.
    in0_memory_config: Any = None


class VisionSdpaPlan(NamedTuple):
    """SDPA config; skip the K cast when it already matches k_dtype."""

    compute_kernel_config: Any
    program_config: Any
    k_dtype: Any
    fidelity: str


_L1_PER_CORE = 1499136  # MEM_L1_SIZE, wormhole/dev_mem_map.h
# Per-core L1 a matmul's CBs and buffers must not use.
_L1_RESERVE = 32 * 1024

# Per-device override, keyed like `_VISION_MM_TUNING_BY_DEVICE`.
_L1_RESERVE_BY_DEVICE = {"N300": 128 * 1024}

# Wormhole only; gated by vision_mm_tuned.
_VISION_MM_TUNING = {
    "patch_embed": dict(in0_l1=False, chunk=5504, in0_block_w=6, fidelity="hifi2", out_l1=False),
    "qkv": dict(in0_l1=False, chunk=1536, in0_block_w=18, fidelity="hifi2", out_l1=False),
    "wo": dict(in0_l1=False, chunk=4096, in0_block_w=24, fidelity="lofi", out_l1=False),
    "mlp_fc1": dict(in0_l1=False, chunk=3072, in0_block_w=6, fidelity="hifi2_fp16", out_l1=False),
    "mlp_fc2": dict(in0_l1=False, chunk=1536, in0_block_w=4, fidelity="hifi2_fp16", out_l1=True),
    "merger_fc1": dict(in0_l1=False, chunk=None, in0_block_w=None, fidelity="hifi2_fp16", out_l1=False),
    "merger_fc2": dict(in0_l1=False, chunk=None, in0_block_w=None, fidelity="hifi2_fp16", out_l1=False),
}

_VISION_MM_TUNING_BY_DEVICE = {
    "T3K": {
        "patch_embed": dict(grid_x=8, in0_block_w=6),
        "qkv": dict(chunk=768, grid_x=8, in0_l1=True, out_l1=True),
        "wo": dict(chunk=3072, fidelity="hifi2", out_l1=True),
        "mlp_fc1": dict(chunk=1536, in0_block_w=18, in0_l1=True, out_l1=True),
        "merger_fc2": dict(chunk=1376, in0_block_w=9, out_l1=True),
    },
}

_UNTUNED_FIDELITY_OP = {"qkv": OpGroup.LI_QKV_PREFILL, "wo": OpGroup.LI_O_PREFILL}
_FIDELITY_NAMES = ("lofi", "hifi2", "hifi2_na", "hifi2_fp16", "hifi2_nol1acc", "hifi4", "hifi4_fp16", "hifi4_fp32")

_VISION_SDPA_TUNING = dict(fidelity="hifi2", k_bf8b=True, q_chunk=128, k_chunk=512, exp_approx=False)

_TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
    ttnn.float32: 4096,
}


def _divisors(n, hi=None):
    return [d for d in range(1, (n if hi is None else min(n, hi)) + 1) if n % d == 0]


def _grid_extent(tiles, max_extent):
    """Largest divisor of the tile count that keeps most of the axis, else the full extent."""
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

        self.vision_mm_tuned = is_wormhole_b0() and os.environ.get("QWEN36_VISION_MM_TUNING", "1") != "0"
        self.vision_ccl_kwargs = _vision_ccl_kwargs(self.device_name)
        self._l1_reserve = _L1_RESERVE_BY_DEVICE.get(self.device_name, _L1_RESERVE)
        if not self.vision_mm_tuned:
            logger.info(
                f"vision matmul tuning is Wormhole-only; {self.arch_name} keeps the untuned config "
                f"(re-sweep with tests/perf/test_sweep_vision_matmuls.py to tune it)"
            )

        self._vision_mm_plans = {}
        self._vision_sdpa_plans = {}

        assert self.n_kv_heads % self.cluster_shape[1] == 0, "n_kv_heads must be divisible by num_devices"

        # Sanity-check the divisibility requirements that the TP code relies on.
        tp = self.cluster_shape[1]
        assert self.n_heads % tp == 0, f"vision n_heads ({self.n_heads}) must be divisible by TP={tp}"
        assert self.qkv_size % tp == 0, f"vision qkv_size ({self.qkv_size}) must be divisible by TP={tp}"
        assert self.dim % tp == 0, f"vision dim ({self.dim}) must be divisible by TP={tp}"
        # Replicate activations unless dim splits into a whole number of tiles per device.
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
        """2D matmul plan; fold fused_activation here, not via linear(activation=)."""
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
        # Not an assert: a non-tile-aligned shape cannot use the 2D plan.
        if rows % tile or k % tile or n % tile:
            logger.debug(f"vision {family}: {rows}x{k}x{n} not tile-aligned -> ttnn auto config")
            return auto
        k_t, n_t = k // tile, n // tile
        grid = self.mesh_device.compute_with_storage_grid_size()

        # Largest chunk at or below the cap that divides rows; None means do not chunk.
        chunk_cap = tune["chunk"] or rows
        chunk = max(c * tile for c in _divisors(rows // tile, max(1, chunk_cap // tile)))
        m_t = chunk // tile

        gx = min(tune.get("grid_x") or _grid_extent(n_t, grid.x), grid.x)
        gy = min(tune.get("grid_y") or _grid_extent(m_t, grid.y), grid.y)
        per_core_m, per_core_n = math.ceil(m_t / gy), math.ceil(n_t / gx)
        cap = 4 if ckc.fp32_dest_acc_en else 8  # DST capacity; fp32 accumulate halves it

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
            # Intermediate CB aliases the output CB for bf16 without fp32 accumulate.
            if ckc.fp32_dest_acc_en:
                interm = per_core_m * per_core_n * 4096
            elif out_dtype is ttnn.bfloat16:
                interm = 0
            else:
                interm = per_core_m * per_core_n * _TILE_BYTES[out_dtype]
            return in0 + in1 + out + interm

        # Largest in0_block_w that divides K and fits L1; if only 1 divides, use all of K.
        candidates = _divisors(k_t, tune["in0_block_w"])
        if candidates == [1] and k_t > 1:
            candidates = [k_t]
        in0_block_w = None
        for cand in sorted(candidates, reverse=True):
            if cb_bytes(cand) <= _L1_PER_CORE - self._l1_reserve:
                in0_block_w = cand
                break
        if in0_block_w is None:
            logger.info(f"vision {family}: {rows}x{k}x{n} has no L1-legal 2D config, leaving it on auto")
            return auto

        # Spend leftover L1 on in0 first, then the output; L1 buffers are paged across the grid.
        free_l1 = (_L1_PER_CORE - self._l1_reserve - cb_bytes(in0_block_w)) * grid.x * grid.y
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

    def vision_sdpa_plan(self, seq_len: int, kv_cache_dtype) -> VisionSdpaPlan:
        """SDPA kernel config; off-arch, k_dtype stays the caller's kv_cache_dtype."""
        cached = self._vision_sdpa_plans.get((seq_len, kv_cache_dtype))
        if cached is not None:
            return cached

        if not self.vision_mm_tuned:
            plan = VisionSdpaPlan(
                compute_kernel_config=self.decoders_optimizations.get_math_fidelity(
                    decoder_id=0, op=OpGroup.SDPA_PREFILL, configuration=self
                ),
                program_config=self.get_attn_sdpa_program_config(Mode.PREFILL, seq_len, None, None),
                k_dtype=kv_cache_dtype,
                fidelity="untuned",
            )
        else:
            tune = dict(_VISION_SDPA_TUNING)
            grid = self.mesh_device.compute_with_storage_grid_size()
            plan = VisionSdpaPlan(
                compute_kernel_config=getattr(self, f"compute_kernel_config_{tune['fidelity']}"),
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(grid.x, grid.y),
                    exp_approx_mode=tune["exp_approx"],
                    q_chunk_size=tune["q_chunk"],
                    k_chunk_size=tune["k_chunk"],
                ),
                k_dtype=ttnn.bfloat8_b if tune["k_bf8b"] else kv_cache_dtype,
                fidelity=tune["fidelity"],
            )
            logger.info(
                f"vision sdpa: seq {seq_len}, grid {grid.x}x{grid.y}, "
                f"q/k chunk {tune['q_chunk']}/{tune['k_chunk']}, {tune['fidelity']}, "
                f"K {'bf8b' if tune['k_bf8b'] else str(kv_cache_dtype)}, exp_approx {tune['exp_approx']}"
            )
        self._vision_sdpa_plans[(seq_len, kv_cache_dtype)] = plan
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
        axis 1, or replicate when dim does not split into whole tiles.
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
