# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar bring-up configuration for the Qwen3-VL copy: bf16, device-derived grids, truncation."""
import math
import os


# Bump whenever the Quasar config changes where or how weights are placed: cached tensors keep their memory config.
WEIGHT_LAYOUT_VERSION = 2  # 2: text weights DRAM interleaved instead of DRAM sharded


def vision_padded_seq_len(n: int) -> int:
    # Always a multiple of 128 (vision_attention). Above 1024 the MLP and WO reshape into 1024-row chunks and,
    # above 2048, the QKV matmul into 2048-row chunks, so anything over 1024 rounds up to a multiple of 2048.
    step = 128 if n <= 1024 else 2048
    return math.ceil(n / step) * step


def truncate_hf_config(config, vision_layers, text_layers, deepstack_at=None):
    config.vision_config.depth = vision_layers
    config.text_config.num_hidden_layers = text_layers
    if deepstack_at is not None:
        config.vision_config.deepstack_visual_indexes = [deepstack_at]
    return config


# --- Quasar model args -------------------------------------------------------------------------
from contextlib import contextmanager  # noqa: E402

import ttnn  # noqa: E402
from models.common.utility_functions import is_quasar  # noqa: E402
from models.experimental.ops.quasar.qwen3_vl.tt.model_config import VisionModelArgs  # noqa: E402
from models.tt_transformers.tt import model_config as _ttt_mc  # noqa: E402
from models.tt_transformers.tt.common import Mode  # noqa: E402
from models.tt_transformers.tt.model_config import (  # noqa: E402
    DecodersPrecision,
    MathFidelitySetting,
    ModelArgs,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)

# ModelArgs keys tuning tables by device name; Quasar has none, so it borrows the single-chip WH entries.
QUASAR_DEVICE_NAME = "N150"


def fp32_dest_acc_requested():
    # A/B switch for hardware runs; fp32 dest accumulation is undefined on ttsim WH (QUASAR_GAPS S1).
    return os.environ.get("QWEN_QSR_FP32_DEST_ACC") == "1"


def bf16_decoders_precision(num_decoders, model_name, fp32_dest_acc=False):
    fidelity = MathFidelitySetting.HIFI4 if fp32_dest_acc else MathFidelitySetting.HIFI4_FP16
    settings = {
        "TensorPrecision": {g: PrecisionSetting.BF16 for g in TensorGroup},
        # Default HiFi4 without fp32 dest accumulation: matmuls reload fp32 partials through srcA, which is undefined.
        "OpFidelity": {g: fidelity for g in OpGroup},
    }
    return DecodersPrecision(num_decoders, model_name, ModelOptimizations(settings))


def strip_fp32_dest_acc(args):
    """Turn off fp32 dest accumulation in every compute_kernel_config_* attribute; returns the names changed."""
    changed = []
    for name, cfg in list(vars(args).items()):
        if name.startswith("compute_kernel_config") and getattr(cfg, "fp32_dest_acc_en", False):
            setattr(
                args,
                name,
                ttnn.WormholeComputeKernelConfig(
                    math_fidelity=cfg.math_fidelity,
                    math_approx_mode=cfg.math_approx_mode,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=cfg.packer_l1_acc,
                ),
            )
            changed.append(name)
    return changed


def fit_matmul_config(cfg, gx, gy, m_tiles=None, n_tiles=None, force=False):
    """Clip a matmul program config to a gx x gy grid; per-core blocks cover the real M/N tiles when given,
    otherwise they grow to keep the original grid's total work."""
    if cfg is None:
        return None
    grid = cfg.compute_with_storage_grid_size
    nx, ny = min(grid.x, gx), min(grid.y, gy)
    if (nx, ny) == (grid.x, grid.y) and not force:
        return cfg
    if isinstance(cfg, ttnn.MinimalMatmulConfig):
        return ttnn.MinimalMatmulConfig(
            M_block_size=cfg.M_block_size,
            K_block_size=cfg.K_block_size,
            N_block_size=cfg.N_block_size,
            compute_with_storage_grid_size=ttnn.CoreCoord(nx, ny),
        )
    if isinstance(cfg, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig):
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(nx, ny),
            in0_block_w=1,  # fewer cores hold bigger output blocks; the smallest K block keeps the CBs in L1
            out_subblock_h=1,  # 1x1 subblocks always divide the rescaled per-core blocks
            out_subblock_w=1,
            per_core_M=math.ceil(m_tiles / ny) if m_tiles else math.ceil(cfg.per_core_M * grid.y / ny),
            per_core_N=math.ceil(n_tiles / nx) if n_tiles else math.ceil(cfg.per_core_N * grid.x / nx),
            transpose_mcast=cfg.transpose_mcast,
            fused_activation=cfg.fused_activation,
            fuse_batch=cfg.fuse_batch,
        )
    raise TypeError(f"fit_matmul_config: unsupported {type(cfg).__name__} on a {gx}x{gy} grid")


@contextmanager
def _quasar_device_name():
    orig = _ttt_mc.determine_device_name

    def name(mesh_device):
        try:
            return orig(mesh_device)
        except ValueError:
            return QUASAR_DEVICE_NAME

    _ttt_mc.determine_device_name = name
    try:
        yield
    finally:
        _ttt_mc.determine_device_name = orig


class _QuasarArgsMixin:
    def _quasar_init(self, parent_init, *args, **kwargs):
        fp32_dest_acc = fp32_dest_acc_requested()
        if kwargs.get("optimizations") is None:
            kwargs["optimizations"] = lambda a: bf16_decoders_precision(a.n_layers, a.model_name, fp32_dest_acc)
        with _quasar_device_name():
            parent_init(self, *args, **kwargs)
        self.lm_head_dtype = ttnn.bfloat16
        self.ccl_dtype = ttnn.bfloat16
        if not fp32_dest_acc:
            strip_fp32_dest_acc(self)

    # --- grid helpers bounded by the real device grid instead of WH's 8x8 ---
    def _grid_bounds(self):
        return self.max_grid_size.y, self.max_grid_size.x

    @staticmethod
    def _rows_cols(cores, max_rows, max_cols):
        for rows in range(1, max_rows + 1):
            if cores % rows == 0 and cores // rows <= max_cols:
                return rows, cores // rows
        return None

    def find_grid(self, N):
        max_rows, max_cols = self._grid_bounds()
        cores = sorted((k for k in range(1, max_rows * max_cols + 1) if N % k == 0), key=lambda k: abs(k - 32))
        for c in cores:
            rc = self._rows_cols(c, max_rows, max_cols)
            if rc:
                return rc
        raise AssertionError(f"no grid for {N} tiles within {max_rows}x{max_cols}")

    def find_grid_k_n(self, K, N):
        max_rows, max_cols = self._grid_bounds()
        for c in sorted((c for c in range(1, max_rows * max_cols + 1) if K % c == 0 and N % c == 0), reverse=True):
            rc = self._rows_cols(c, max_rows, max_cols)
            if rc:
                return rc
        raise AssertionError(f"no grid for K={K}, N={N} within {max_rows}x{max_cols}")

    def find_prefill_grid(self, row_tiles, col_tiles):
        # Returned as (x, y): every consumer (matmul_config, MinimalMatmulConfig) reads it that way. The base class
        # returns (rows, cols), which only works because its grid is a square 8x8.
        max_rows, max_cols = self._grid_bounds()
        cols = next(i for i in range(max_cols, 0, -1) if col_tiles % i == 0)
        rows = next(i for i in range(max_rows, 0, -1) if row_tiles % i == 0)
        return cols, rows

    # --- program and memory configs that hardcode an 8x8 grid ---
    def _device_grid(self):
        return (self.max_grid_size.x, self.max_grid_size.y)

    def get_attn_sdpa_decode_program_config(self, prefetcher=None):
        cfg = super().get_attn_sdpa_decode_program_config(prefetcher)
        cfg.compute_with_storage_grid_size = self._device_grid()
        return cfg

    def get_attn_sdpa_prefill_program_config(self, seq_len=1, chunk_start_idx=None):
        cfg = super().get_attn_sdpa_prefill_program_config(seq_len, chunk_start_idx)
        cfg.compute_with_storage_grid_size = self._device_grid()
        return cfg

    def get_attn_sdpa_output_mem_config(self, mode, batch_size_per_device_group=1, prefetcher=None):
        orig = _ttt_mc.num_to_corerange
        gx, gy = self._device_grid()
        _ttt_mc.num_to_corerange = lambda x, start_core=ttnn.CoreCoord(0, 0), grid_x=8, grid_y=8: orig(
            x, start_core, gx, gy
        )
        try:
            return super().get_attn_sdpa_output_mem_config(mode, batch_size_per_device_group, prefetcher)
        finally:
            _ttt_mc.num_to_corerange = orig

    # Per-call M is the chunk each prefill matmul sees after the model's own reshapes (attention.py / mlp.py).
    def _m_tiles(self, mode, seq_len, chunk):
        return 1 if mode != Mode.PREFILL else math.ceil(min(seq_len, chunk) / ttnn.TILE_SIZE)

    def _fit(self, cfg, mode, seq_len, chunk, n):
        gx, gy = self._device_grid()
        small = gx * gy < 64  # below WH's 8x8 the base configs' K blocks no longer fit in L1: always rebuild
        return fit_matmul_config(cfg, gx, gy, self._m_tiles(mode, seq_len, chunk), n // ttnn.TILE_SIZE, force=small)

    def get_attn_qkv_program_config(self, mode, seq_len=1, prefetcher=None):
        cfg = super().get_attn_qkv_program_config(mode, seq_len, prefetcher)
        return self._fit(cfg, mode, seq_len, getattr(self, "MAX_QKV_MM_SEQ_LEN", 2048), self.qkv_size)

    def get_attn_wo_program_config(self, mode, seq_len=1, prefetcher=None):
        return self._fit(super().get_attn_wo_program_config(mode, seq_len, prefetcher), mode, seq_len, 1024, self.dim)

    def get_mlp_ff1_3_prg_config(self, mode, seq_len=1, prefetcher=None):
        cfg = super().get_mlp_ff1_3_prg_config(mode, seq_len, prefetcher)
        return self._fit(cfg, mode, seq_len, self.prefill_len_cutoff, self.hidden_dim)

    def get_mlp_ff2_prg_config(self, mode, seq_len=1, prefetcher=None):
        cfg = super().get_mlp_ff2_prg_config(mode, seq_len, prefetcher)
        return self._fit(cfg, mode, seq_len, self.prefill_len_cutoff, self.dim)

    # DRAM-sharded weights need one reader core per DRAM bank (12 on WH, more than the emulator's 2 cores), and the
    # Quasar DRAM-sharded matmul is not in yet (#58912): keep weights DRAM interleaved and let ttnn pick the matmul.
    def create_dram_sharded_mem_config(self, k, n, dram_grid=None):
        return ttnn.DRAM_MEMORY_CONFIG

    def dram_matmul_config(self, m, k, n, num_cores=None, fused_activation=None, num_workers_per_dram_bank=1):
        return None

    def _quasar_fix_grid_attrs(self):
        g = self.max_grid_size
        self.dram_shard_grid_width = g.x  # per_core_N of the prefill matmul configs assumes this many columns
        rows, cols = self.find_grid(self.dim // ttnn.TILE_SIZE)
        self.lm_head_core_grid = ttnn.CoreGrid(y=rows, x=cols)
        per = ttnn.TILE_SIZE * self.lm_head_core_grid.num_cores
        budget = self.get_lm_head_max_columns_per_device(self.lm_head_core_grid, self.prefetcher)
        self.max_columns_per_device_lm_head = max(per, budget // per * per)
        self.min_kv_prefill_shard_seqlen = float("inf")  # never L1-shard K/V for fill_cache on small grids
        self.model_config["LM_HEAD_OUTPUT_MEMCFG"] = ttnn.DRAM_MEMORY_CONFIG


class QuasarModelArgs(_QuasarArgsMixin, ModelArgs):
    def __init__(self, *args, **kwargs):
        self._quasar_init(ModelArgs.__init__, *args, **kwargs)
        self._quasar_fix_grid_attrs()
        # ttnn.scatter is not ported to Quasar; the vision-token merge copies rows on the host instead.
        self.device_scatter = False


class QuasarVisionModelArgs(_QuasarArgsMixin, VisionModelArgs):
    def __init__(self, *args, **kwargs):
        self._quasar_init(VisionModelArgs.__init__, *args, **kwargs)
        self.vision_weight_dtype = ttnn.bfloat16
        self.vision_mlp_fc1_dtype = ttnn.bfloat16


def model_args_classes(force=False):
    if force or is_quasar():
        return QuasarModelArgs, QuasarVisionModelArgs
    return ModelArgs, VisionModelArgs
