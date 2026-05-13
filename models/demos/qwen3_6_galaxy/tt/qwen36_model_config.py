# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B model configuration for TT Galaxy (BH GLX 8×4 mesh, 32 chips).

This is a STANDALONE class — it does NOT extend TtQwenModelArgs (Qwen3-32B) so the
two configs are fully independent and cannot regress each other.

Design: pad-and-slice strategy
-------------------------------
Qwen3.6-27B has smaller Q/KV head counts and a smaller intermediate dim than Qwen3-32B.
To reuse the Qwen3-32B matmul program configs (which are tuned for specific shapes) we
pad the native counts up to the Qwen3-32B values and slice the outputs back to native
size after every matmul.  The patterns here follow:
  - models/demos/llama3_70b_galaxy/tt/qwen_model_config.py  (Qwen3-32B on same HW)
  - models/demos/olmo_galaxy/tt/olmo_model_config.py         (Olmo-3.1-32B, same pad strategy)

Padding strategy
-----------------
Native dims (from HF config.json text_config):
    hidden_size          = 5120   (dim_per_tp = 1280 after /4 — already tile-aligned, no pad)
    num_attention_heads  = 24     (n_q per col = 6  → pad to 16 per col → n_q_padded = 64)
    num_key_value_heads  = 4      (n_kv per col = 1 → pad to 2 per col  → n_kv_padded =  8)
    head_dim             = 256    (keep as-is; re-derive tile shapes using 256, not 128)
    intermediate_size    = 17408  (per-tp = 2176 → pad to 3840, 24-core aligned)
    vocab_size           = 248320 (pad to 248832 = 32×7776, multiple of 1024)

Qwen3.6-specific features (absent in Qwen3-32B):
    - attn_output_gate=True (output gate on full attention, swish-gated)
    - partial_rotary_factor=0.25 → rope_dim=64 (MRoPE on first 64 of 256 head dims)
    - mrope_section=[11,11,10] partitions rotary_dim=64 into t/h/w groups
    - layer_types: 64 entries, [lin,lin,lin,full]×16 (GatedDeltaNet + full attn)
    - linear_num_key_heads=16, linear_num_value_heads=48, linear_head_dim=128
    - linear_conv_kernel=4 (short causal conv in GatedDeltaNet)

TODO (later tasks)
-------------------
- Sharded memcfgs for QKV / FF projections (head_dim=256 vs 128 changes shard shapes)
- Prog configs for padded intermediate sizes
- KV-cache sharded memcfgs
- Prefill SDPA prog config
- Embedding / LM-head sharded memcfg (padded_vocab_size = 248832)
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole

# ---------------------------------------------------------------------------
# Paged attention configuration
# ---------------------------------------------------------------------------


@dataclass
class PagedAttentionConfig:
    """Configuration for paged KV cache.

    Parameters
    ----------
    block_size : int
        Number of tokens per physical page block (must be a multiple of 32 for
        tile-alignment in TILE_LAYOUT).  Typical values: 32, 64, 128.
    max_num_blocks : int
        Total number of physical page blocks allocated in the paged KV cache.
        Per-device allocation: each chip in a column holds
            max_num_blocks × n_kv_per_col × block_size × head_dim tokens.
        Size for unit tests: max_seq_len × max_batch_size / block_size (rounded up).
    """

    block_size: int = 64
    max_num_blocks: int = 8  # Default: enough for 512-token sequence at block_size=64


# ---------------------------------------------------------------------------
# Padding constants (mirroring qwen_model_config.py / olmo_model_config.py)
# ---------------------------------------------------------------------------

# These are the "target" padded dims that the existing Qwen3-32B program configs
# were written for. We pad Qwen3.6-27B's smaller counts up to these numbers and
# slice the results back to the native sizes afterwards.
_N_Q_HEADS_PADDED = 64  # Qwen3-32B n_q_heads — target for pad-and-slice
_N_KV_HEADS_PADDED = 8  # Qwen3-32B n_kv_heads — target for pad-and-slice
_INTERMEDIATE_PER_TP_PADDED_24_CORES = 3840  # 24-core aligned (same as Qwen3-32B & Olmo)
_DIM_PADDED_24_CORES = 6144  # same as Qwen3-32B


class TtQwen36ModelArgs:
    """Qwen3.6-27B model configuration for BH GLX (8×4 mesh, 32 chips).

    Can be constructed with ``mesh_device=None`` for offline/CPU-only tests;
    hardware-derived attributes (cluster_shape, device_name, CCL_TOPOLOGY, etc.)
    are skipped in that case.

    Parameters
    ----------
    mesh_device: ttnn.MeshDevice | None
        The full 8×4 mesh device.  Pass ``None`` for offline use.
    instruct: bool
        Whether to use the instruct variant (affects tokenizer prompt template).
    max_batch_size: int
        Maximum decode batch size.
    max_seq_len: int
        Maximum sequence length for KV-cache allocation.
    attn_output_gate: bool
        Override for the output gate flag (default True, matching HF config).
        Exposed here to allow test_qkv_size_includes_gate to flip it.
    """

    # ---------------------------------------------------------------------------
    # Tensor-parallel factors
    # ---------------------------------------------------------------------------
    dim_tp_factor = 4  # col dimension (mesh cols = 4)
    intermediate_dim_tp_factor = 8  # row × col (mesh rows=8, mesh cols=4 both used)

    # ---------------------------------------------------------------------------
    # BH GLX specific
    # ---------------------------------------------------------------------------
    # (set in __init__ based on is_blackhole())
    GALAXY_NUM_LINKS: int
    CCL_TOPOLOGY: ttnn.Topology
    device_name: str

    def __init__(
        self,
        mesh_device,
        instruct: bool = False,
        max_batch_size: int = 1,
        max_seq_len: int = 128 * 1024,
        attn_output_gate: Optional[bool] = None,
        use_paged_kv_cache: bool = False,
        block_size: int = 64,
        max_num_blocks: Optional[int] = None,
    ):
        # ---------------------------------------------------------------
        # 1. Device bookkeeping
        # ---------------------------------------------------------------
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices() if mesh_device is not None else 0
        self.tile_size = 32
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.instruct = instruct
        self.tile_padded_batch_rows = self.tile_size * int(math.ceil(max_batch_size / self.tile_size))

        # ---------------------------------------------------------------
        # 2. Validate mesh shape
        # ---------------------------------------------------------------
        if self.num_devices not in (0, 32):
            raise ValueError(
                f"TtQwen36ModelArgs: unsupported num_devices={self.num_devices}. "
                "Only 32 devices (Galaxy 8×4) or 0 (offline/CPU) are supported."
            )

        # ---------------------------------------------------------------
        # 3. Native model dimensions (from HF config.json text_config)
        # ---------------------------------------------------------------
        self.dim = 5120
        self.n_heads = 24  # num_attention_heads (Q heads)
        self.n_q_heads_native = 24  # alias kept for pad-and-slice clarity
        self.n_kv_heads = 4  # num_key_value_heads
        self.n_kv_heads_native = 4  # alias
        self.head_dim = 256  # explicit in config (NOT dim//n_heads=213)
        self.num_hidden_layers = 64
        self.vocab_size = 248320
        self.vocab_size_native = 248320  # alias
        self.norm_eps = 1e-6
        self.intermediate_dim = 17408  # intermediate_size
        self.intermediate_dim_native = 17408  # alias
        self.n_layers = self.num_hidden_layers  # some consumers use n_layers

        # ---------------------------------------------------------------
        # 4. Pad-and-slice targets
        # ---------------------------------------------------------------
        # Q heads: 24 → 64 (existing 64-head shard programs apply)
        self.n_q_heads_padded = _N_Q_HEADS_PADDED  # 64
        # KV heads: 4 → 8
        self.n_kv_heads_padded = _N_KV_HEADS_PADDED  # 8

        # Intermediate: 17408/8=2176 → pad to 3840 (24-core aligned)
        self.intermediate_dim_per_tp_native = self.intermediate_dim // self.intermediate_dim_tp_factor  # 2176
        assert (
            self.intermediate_dim_per_tp_native == 2176
        ), f"Expected intermediate_dim_per_tp_native=2176, got {self.intermediate_dim_per_tp_native}"
        self.intermediate_dim_per_tp_padded = _INTERMEDIATE_PER_TP_PADDED_24_CORES  # 3840
        # Convenience alias used by some downstream blocks
        self.intermediate_dim_per_tp_padded_24_cores = _INTERMEDIATE_PER_TP_PADDED_24_CORES

        # Vocab: 248320 → 248832 (= 32 × 7776 = ceil(248320/1024)*1024)
        _padded = math.ceil(self.vocab_size_native / 1024) * 1024
        assert _padded == 248832, f"Unexpected padded_vocab_size={_padded}"
        self.padded_vocab_size = _padded  # 248832

        # dim_per_tp: 5120 / 4 = 1280 (already tile-aligned — no padding needed)
        self.dim_per_tp = self.dim // self.dim_tp_factor  # 1280
        assert self.dim_per_tp == 1280 and self.dim_per_tp % 32 == 0, f"dim_per_tp={self.dim_per_tp} not tile-aligned"

        # dim_padded_24_cores: kept for compatibility with downstream blocks that
        # query this attribute (same value as Qwen3-32B, since dim=5120 is shared)
        self.dim_padded_24_cores = _DIM_PADDED_24_CORES  # 6144

        # ---------------------------------------------------------------
        # 5. Qwen3.6-specific flags
        # ---------------------------------------------------------------
        # attn_output_gate: override from constructor or default True (from HF)
        if attn_output_gate is None:
            self.attn_output_gate = True
        else:
            self.attn_output_gate = bool(attn_output_gate)

        self.partial_rotary_factor: float = 0.25
        self.rope_dim: int = int(self.head_dim * self.partial_rotary_factor)  # 64
        self.mrope_section: list = [11, 11, 10]  # partitions rotary_dim=64 into t/h/w
        self.mrope_theta: float = 10_000_000.0

        # Verify mrope_section sums to half of rope_dim
        assert sum(self.mrope_section) == self.rope_dim // 2, (
            f"mrope_section {self.mrope_section} sums to {sum(self.mrope_section)}, " f"expected {self.rope_dim // 2}"
        )

        self.qk_norm: bool = True
        self.qk_norm_zero_centered: bool = True  # q_norm/k_norm use (1+w)*x convention
        self.input_norm_zero_centered: bool = True  # input/post-attn/final norms same
        self.deltanet_norm_zero_centered: bool = False  # GatedDeltaNet inner norm: standard

        # Linear attention (GatedDeltaNet) parameters
        self.linear_num_key_heads: int = 16
        self.linear_num_value_heads: int = 48
        self.linear_head_dim: int = 128  # shared key/value head dim
        self.linear_conv_kernel: int = 4

        # Layer type pattern (64 entries, [lin, lin, lin, full] × 16)
        # Loaded from HF config if available, otherwise hardcoded
        self.linear_attention_pattern: list = self._load_layer_types()

        # ---------------------------------------------------------------
        # 6. qkv_size property (computed attribute)
        # ---------------------------------------------------------------
        # Set as a plain attribute here; the @property variant is defined below
        # but we also expose a flat attribute for simpler downstream access.
        # See qkv_size property.

        # ---------------------------------------------------------------
        # 7. Legacy / compatibility aliases
        # ---------------------------------------------------------------
        self.is_qwen = True
        self.is_galaxy = self.num_devices == 32

        # ---------------------------------------------------------------
        # 8. Paged KV cache configuration
        # ---------------------------------------------------------------
        self.use_paged_kv_cache: bool = use_paged_kv_cache
        if use_paged_kv_cache:
            # Compute max_num_blocks if not provided:
            # enough for max_seq_len × max_batch_size tokens at the given block_size.
            if max_num_blocks is None:
                import math as _math

                _max_num_blocks = _math.ceil(max_seq_len / block_size) * max_batch_size
            else:
                _max_num_blocks = max_num_blocks
            self.paged_attention_config = PagedAttentionConfig(
                block_size=block_size,
                max_num_blocks=_max_num_blocks,
            )
        else:
            self.paged_attention_config = None

        # ---------------------------------------------------------------
        # 8. Hardware-derived attributes (skipped when mesh_device is None)
        # ---------------------------------------------------------------
        if mesh_device is not None:
            self._init_hardware_attrs(mesh_device)
        else:
            # Offline mode — set hardware attrs to sentinel values
            self.cluster_shape = None
            self.device_name = "CPU"
            self.GALAXY_NUM_LINKS = None
            self.CCL_TOPOLOGY = None
            self.n_local_heads = None
            self.n_local_kv_heads = None

    # ------------------------------------------------------------------
    # Hardware initialisation (separated to keep __init__ readable)
    # ------------------------------------------------------------------

    def _init_hardware_attrs(self, mesh_device):
        """Initialise all attributes that require the mesh device."""
        self.cluster_shape = list(mesh_device.shape)  # [8, 4]

        # Verify cluster shape
        assert self.cluster_shape == [8, 4], f"Expected cluster_shape=[8,4] for BH GLX, got {self.cluster_shape}"
        assert self.num_devices == 32, f"Expected 32 devices for BH GLX Galaxy, got {self.num_devices}"

        # BH vs WH device name and CCL topology
        self.device_name = "BH_GLX" if is_blackhole() else "TG"
        self.GALAXY_NUM_LINKS = 1 if is_blackhole() else 4
        self.CCL_TOPOLOGY = ttnn.Topology.Linear if is_blackhole() else ttnn.Topology.Ring

        # Per-device head counts (native)
        n_cols = self.cluster_shape[1]  # 4
        assert self.n_heads % n_cols == 0, f"n_heads={self.n_heads} must be divisible by cluster cols={n_cols}"
        assert self.n_kv_heads % n_cols == 0, f"n_kv_heads={self.n_kv_heads} must be divisible by cluster cols={n_cols}"
        self.n_local_heads = self.n_heads // n_cols  # 24/4 = 6 (native)
        self.n_local_kv_heads = self.n_kv_heads // n_cols  # 4/4  = 1 (native)

        # Padded per-column counts
        self.n_local_heads_padded = self.n_q_heads_padded // n_cols  # 64/4 = 16
        self.n_local_kv_heads_padded = self.n_kv_heads_padded // n_cols  # 8/4  = 2

        # Device group accounting (for Galaxy KV-cache replication)
        self.TG = True
        self.num_device_groups = self.num_devices // self.n_kv_heads_padded
        self.num_devices_per_group = self.n_kv_heads_padded

        logger.info(
            f"TtQwen36ModelArgs: device_name={self.device_name}, "
            f"cluster_shape={self.cluster_shape}, "
            f"CCL_TOPOLOGY={self.CCL_TOPOLOGY}, "
            f"GALAXY_NUM_LINKS={self.GALAXY_NUM_LINKS}"
        )

    # ------------------------------------------------------------------
    # Layer-type pattern loader
    # ------------------------------------------------------------------

    def _load_layer_types(self) -> list:
        """Load layer_types from HF config.json or return the hardcoded pattern.

        Returns the 64-entry list from the HF text_config['layer_types'].  If the
        HF snapshot is not available, falls back to the hardcoded [lin,lin,lin,full]×16
        pattern which matches the config exactly.
        """
        _snapshot = Path(
            "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
            "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
        )
        cfg_path = _snapshot / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                return cfg["text_config"]["layer_types"]
            except Exception as e:
                logger.warning(f"Could not load layer_types from HF config: {e}")

        # Fallback: hardcoded [lin, lin, lin, full] × 16
        pattern = (["linear_attention", "linear_attention", "linear_attention", "full_attention"]) * 16
        logger.info("Using hardcoded layer_types pattern (HF config not found)")
        return pattern

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def qkv_size(self) -> int:
        """Total size of the packed QKV (+ output gate Q) projection output.

        When attn_output_gate=True (default, Qwen3.6 uses a Q gate):
            output = [Q | K | V | Q_gate]
            size   = head_dim × (n_kv + n_kv + n_q + n_q)
                   = head_dim × (2 × n_kv_heads + 2 × n_heads)

        When attn_output_gate=False (standard GQA):
            output = [Q | K | V]
            size   = head_dim × (n_q + 2 × n_kv)
        """
        if self.attn_output_gate:
            return self.head_dim * (2 * self.n_kv_heads + 2 * self.n_heads)
        else:
            return self.head_dim * (2 * self.n_kv_heads + self.n_heads)

    # ------------------------------------------------------------------
    # Helper methods for downstream blocks
    # ------------------------------------------------------------------

    def head_count_per_col(self, padded: bool = True):
        """Return (n_q_per_col, n_kv_per_col) for a single mesh column.

        Parameters
        ----------
        padded: bool
            If True (default), return the padded counts (for program-config selection).
            If False, return the native counts (for output slicing).
        """
        n_cols = self.cluster_shape[1] if self.cluster_shape is not None else 4
        if padded:
            return (self.n_q_heads_padded // n_cols, self.n_kv_heads_padded // n_cols)
        else:
            return (self.n_heads // n_cols, self.n_kv_heads // n_cols)

    def intermediate_per_row(self, padded: bool = True) -> int:
        """Return the intermediate_dim split per mesh row (TP axis).

        Parameters
        ----------
        padded: bool
            If True (default), return padded 3840; else return native 2176.
        """
        if padded:
            return self.intermediate_dim_per_tp_padded
        else:
            return self.intermediate_dim_per_tp_native

    def vocab_per_chip(self) -> int:
        """Return vocab shard size per chip (padded_vocab_size / num_devices).

        248832 / 32 = 7776 tokens per chip.
        """
        return self.padded_vocab_size // 32  # 7776

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TtQwen36ModelArgs(\n"
            f"  dim={self.dim}, n_heads={self.n_heads}, head_dim={self.head_dim},\n"
            f"  n_kv_heads={self.n_kv_heads}, intermediate_dim={self.intermediate_dim},\n"
            f"  n_q_heads_padded={self.n_q_heads_padded}, n_kv_heads_padded={self.n_kv_heads_padded},\n"
            f"  intermediate_dim_per_tp_padded={self.intermediate_dim_per_tp_padded},\n"
            f"  padded_vocab_size={self.padded_vocab_size},\n"
            f"  attn_output_gate={self.attn_output_gate}, qkv_size={self.qkv_size},\n"
            f"  rope_dim={self.rope_dim}, mrope_section={self.mrope_section},\n"
            f"  device_name={self.device_name}, num_devices={self.num_devices},\n"
            f"  cluster_shape={self.cluster_shape},\n"
            f"  CCL_TOPOLOGY={self.CCL_TOPOLOGY}, GALAXY_NUM_LINKS={self.GALAXY_NUM_LINKS}\n"
            f")"
        )
