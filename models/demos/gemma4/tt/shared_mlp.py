# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Shared/Dense MLP with GeGLU activation.

Each decoder layer has a shared MLP (plus routed MoE experts on MoE variants).
Architecture: down_proj(GELU(gate_proj(x)) * up_proj(x)), no bias.

HF weight shapes (Gemma4-31B, hidden_size=5376, intermediate_size=21504):
  gate_proj.weight: [intermediate_size, hidden_size]
  up_proj.weight:   [intermediate_size, hidden_size]
  down_proj.weight: [hidden_size, intermediate_size]

Decode matmul backends (mutually exclusive):
  1. Interleaved ``ttnn.linear`` — prefill + default decode.
  2. ``GEMMA4_DRAM_SHARDED`` — DRAM-width-sharded peak-BW matmul (Phase 2a).
  3. ``GEMMA4_PREFETCHER`` — ring-1D matmul + Prefetcher global CB (Phase 2b).
"""

import math
import os

from loguru import logger

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.demos.gemma4.utils.general_utils import get_cache_file_name
from models.tt_transformers.tt.common import pad_to_size

TILE = 32


def _env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _find_grid_k_n(k_tiles, n_tiles, max_rows=8, max_cols=8):
    """Largest core-grid (rows, cols) whose core count divides both k_tiles and n_tiles."""
    max_cores = max_rows * max_cols
    possible = [c for c in range(1, max_cores + 1) if k_tiles % c == 0 and n_tiles % c == 0]
    possible.sort(reverse=True)
    for cores in possible:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols
    raise AssertionError(f"No core grid divides both k_tiles={k_tiles} and n_tiles={n_tiles}")


def _largest_divisor(n, max_divisor=8):
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def _pf_pad_width(tensor, target_w: int, worker_grids):
    """Pad last dim to ``target_w`` without spanning Prefetcher sender cores.

    TILE ``ttnn.pad`` routes through ``fill_pad``, which ignores
    ``sub_core_grids`` and fails under dual sub-devices. Untilize → pad →
    ``tilize_with_val_padding`` (H=1 → 32) keeps work on the worker rectangle.
    """
    if tensor.shape[-1] >= target_w and tensor.layout == ttnn.TILE_LAYOUT:
        return tensor
    pad_amt = max(0, target_w - tensor.shape[-1])
    rm = (
        tensor
        if tensor.layout == ttnn.ROW_MAJOR_LAYOUT
        else ttnn.untilize(tensor, use_multicore=True, sub_core_grids=worker_grids)
    )
    if pad_amt > 0:
        padded = ttnn.pad(
            rm,
            [(0, 0), (0, 0), (0, 0), (0, pad_amt)],
            value=0.0,
            sub_core_grids=worker_grids,
        )
        if rm is not tensor:
            rm.deallocate(True)
        rm = padded
    # Decode activations are H=1; tilize requires H % TILE == 0.
    s = rm.shape
    out_shape = ttnn.Shape([s[0], s[1], TILE, s[3]])
    out = ttnn.tilize_with_val_padding(
        rm,
        out_shape,
        0.0,
        use_multicore=True,
        sub_core_grids=worker_grids,
    )
    if rm is not tensor:
        rm.deallocate(True)
    return out


def _pf_slice_width(tensor, logical_w: int, worker_grids):
    """Crop last dim to ``logical_w`` on Prefetcher worker cores."""
    if tensor.shape[-1] <= logical_w:
        return tensor
    s = tensor.shape
    return ttnn.slice(
        tensor,
        [0, 0, 0, 0],
        [s[0], s[1], s[2], logical_w],
        sub_core_grids=worker_grids,
    )


class SharedMLP:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        mesh_config,
        ccl_manager=None,
        dtype=ttnn.bfloat8_b,
        down_dtype=None,
        tensor_cache_path=None,
        prefetcher=None,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.prefetcher = prefetcher
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size

        if down_dtype is None:
            down_dtype = dtype

        tp = mesh_config.tp if mesh_config else 1
        self.tp = tp
        tp_suffix = f"_tp{tp}" if tp > 1 else ""

        from models.demos.gemma4.tt.precision import dtype_to_str

        dtype_suffix = f"_{dtype_to_str(dtype)}"
        down_dtype_suffix = f"_{dtype_to_str(down_dtype)}"

        if tp > 1:
            col_mapper = mesh_config.column_parallel(mesh_device)
            row_mapper = mesh_config.row_parallel(mesh_device)
        else:
            col_mapper = None
            row_mapper = None

        # Prefetcher and DRAM-sharded are mutually exclusive on the same matmul.
        self.use_prefetcher = prefetcher is not None
        self.use_dram_sharded = False
        if self.use_prefetcher:
            pass  # ring path below
        elif _env_bool("GEMMA4_DRAM_SHARDED_MLP") or _env_bool("GEMMA4_DRAM_SHARDED"):
            try:
                self._setup_dram_sharded(mesh_device, tp)
                self.use_dram_sharded = True
                logger.info(
                    "SharedMLP DRAM-sharded decode matmuls enabled "
                    f"(gate/up cores={self._gate_up_cores}, down cores={self._down_cores}, "
                    f"dram_banks={self._dram_cores})"
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"SharedMLP DRAM-sharded setup failed ({e}); using interleaved matmuls")
                self.use_dram_sharded = False

        if state_dict:
            gate_proj_weight = state_dict["gate_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            up_proj_weight = state_dict["up_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            down_proj_weight = state_dict["down_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        else:
            gate_proj_weight = None
            up_proj_weight = None
            down_proj_weight = None

        if self.use_prefetcher:
            self._setup_prefetcher_weights(
                mesh_device,
                tp,
                gate_proj_weight,
                up_proj_weight,
                down_proj_weight,
                col_mapper,
                row_mapper,
                dtype,
                down_dtype,
                tensor_cache_path,
                tp_suffix,
                dtype_suffix,
                down_dtype_suffix,
            )
        else:
            # Interleaved weights (prefill + decode baseline / DRAM-sharded seed).
            self.gate_proj = ttnn.as_tensor(
                gate_proj_weight,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=col_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_proj.weight{tp_suffix}{dtype_suffix}"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.up_proj = ttnn.as_tensor(
                up_proj_weight,
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=col_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, f"up_proj.weight{tp_suffix}{dtype_suffix}"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.down_proj = ttnn.as_tensor(
                down_proj_weight,
                device=mesh_device,
                dtype=down_dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=row_mapper,
                cache_file_name=get_cache_file_name(
                    tensor_cache_path, f"down_proj.weight{tp_suffix}{down_dtype_suffix}"
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            self._gate_proj_ds = self._up_proj_ds = self._down_proj_ds = None
            if self.use_dram_sharded:
                try:
                    self._gate_proj_ds = ttnn.to_memory_config(self.gate_proj, self._gate_up_weight_mem_config)
                    self._up_proj_ds = ttnn.to_memory_config(self.up_proj, self._gate_up_weight_mem_config)
                    self._down_proj_ds = ttnn.to_memory_config(self.down_proj, self._down_weight_mem_config)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"SharedMLP could not allocate DRAM-sharded weight copies ({e}); decode uses interleaved"
                    )
                    for _t in (self._gate_proj_ds, self._up_proj_ds, self._down_proj_ds):
                        if _t is not None:
                            _t.deallocate(True)
                    self._gate_proj_ds = self._up_proj_ds = self._down_proj_ds = None
                    self.use_dram_sharded = False

    def _setup_prefetcher_weights(
        self,
        mesh_device,
        tp,
        gate_w,
        up_w,
        down_w,
        col_mapper,
        row_mapper,
        dtype,
        down_dtype,
        tensor_cache_path,
        tp_suffix,
        dtype_suffix,
        down_dtype_suffix,
    ):
        """Load ring-padded DRAM-width-sharded weights and register with Prefetcher."""
        from models.demos.gemma4.tt import prefetcher_ring as pr

        pf = self.prefetcher
        ring = pf.ring_size
        dram_cores = mesh_device.dram_grid_size().x
        assert mesh_device.dram_grid_size().y == 1

        hidden = self.hidden_size
        inter_per_dev = self.intermediate_size // tp
        assert self.intermediate_size % tp == 0

        # Ring pad every width that is width-sharded across ring_size cores.
        # Gemma4 hidden=5376 → 168 tiles; 168 % 64 ≠ 0, so pad 5376→6144
        # (unlike peers where dim % ring_size is already tile-aligned).
        n_pad = pr.pad_n_to_ring_size(inter_per_dev, ring)  # 5376 → 6144
        k_pad_down = pr.pad_n_to_ring_size(inter_per_dev, ring)  # same for down K
        k_pad_hidden = pr.pad_n_to_ring_size(hidden, ring)  # 5376 → 6144
        n_pad_down = k_pad_hidden  # down output N must also be ring-aligned

        # Prefill still needs unpadded interleaved weights (plain 2D matmul).
        # Keep thin interleaved copies for seq_len > 32; decode uses ring copies.
        self.gate_proj = ttnn.as_tensor(
            gate_w,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_proj.weight{tp_suffix}{dtype_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.up_proj = ttnn.as_tensor(
            up_w,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"up_proj.weight{tp_suffix}{dtype_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.down_proj = ttnn.as_tensor(
            down_w,
            device=mesh_device,
            dtype=down_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=row_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj.weight{tp_suffix}{down_dtype_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Ring-padded DRAM-sharded decode weights (4D required by dram_prefetcher).
        # Pad the *full* host tensor so each TP shard lands at n_pad / k_pad_down
        # (peer test_prefetcher_BH: n_padded * num_devices for N-sharded).
        n_pad_full = n_pad * tp
        k_pad_down_full = k_pad_down * tp
        # gate/up: [1,1,hidden,inter] — pad K (hidden) and N (inter).
        gate_pad = gate_w
        up_pad = up_w
        if gate_pad is not None:
            gate_pad = pad_to_size(gate_pad, dim=-2, size=k_pad_hidden)
            gate_pad = pad_to_size(gate_pad, dim=-1, size=n_pad_full)
        if up_pad is not None:
            up_pad = pad_to_size(up_pad, dim=-2, size=k_pad_hidden)
            up_pad = pad_to_size(up_pad, dim=-1, size=n_pad_full)
        # down: [1,1,inter,hidden] — pad K (inter) and N (hidden).
        down_pad = down_w
        if down_pad is not None:
            down_pad = pad_to_size(down_pad, dim=-2, size=k_pad_down_full)
            down_pad = pad_to_size(down_pad, dim=-1, size=n_pad_down)

        gate_mem = pr.create_dram_sharded_mem_config(k_pad_hidden, n_pad, dram_cores)
        up_mem = gate_mem
        down_mem = pr.create_dram_sharded_mem_config(k_pad_down, n_pad_down, dram_cores)

        ring_suffix = f"_ring{ring}_n{n_pad}_k{k_pad_hidden}"
        self._gate_proj_pf = ttnn.as_tensor(
            gate_pad,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(
                tensor_cache_path, f"gate_proj.weight{tp_suffix}{dtype_suffix}{ring_suffix}"
            ),
            memory_config=gate_mem,
        )
        self._up_proj_pf = ttnn.as_tensor(
            up_pad,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(
                tensor_cache_path, f"up_proj.weight{tp_suffix}{dtype_suffix}{ring_suffix}"
            ),
            memory_config=up_mem,
        )
        self._down_proj_pf = ttnn.as_tensor(
            down_pad,
            device=mesh_device,
            dtype=down_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=row_mapper,
            cache_file_name=get_cache_file_name(
                tensor_cache_path, f"down_proj.weight{tp_suffix}{down_dtype_suffix}{ring_suffix}"
            ),
            memory_config=down_mem,
        )

        # Program / activation configs for decode ring matmuls.
        # Use core_config — Prefetcher.receiver_cores is only bound after init(),
        # which runs later in switch_mode(DECODE).
        receivers = pf.to_core_range_set(pf.core_config.receiver_cores(sender_active=True, receiver_active=True))
        self._pf_ring = ring
        self._pf_n_pad = n_pad
        self._pf_k_pad_down = k_pad_down
        self._pf_k_pad_hidden = k_pad_hidden
        self._pf_n_pad_down = n_pad_down
        self._pf_gate_pc = pr.matmul_1d_ring_config(TILE, k_pad_hidden, n_pad, ring, pf.num_receiver_cores)
        self._pf_up_pc = self._pf_gate_pc
        self._pf_down_pc = pr.matmul_1d_ring_config(TILE, k_pad_down, n_pad_down, ring, pf.num_receiver_cores)
        self._pf_in_mem = pr.activation_mem_config(k_pad_hidden, ring, receivers)
        self._pf_gate_out_mem = pr.output_mem_config(n_pad, ring, receivers)
        self._pf_down_in_mem = pr.activation_mem_config(k_pad_down, ring, receivers)
        self._pf_down_out_mem = pr.output_mem_config(n_pad_down, ring, receivers)

        def register_weights():
            pf.insert_tensor(self._gate_proj_pf)
            pf.insert_tensor(self._up_proj_pf)
            pf.insert_tensor(self._down_proj_pf)

        pf.register_callback(register_weights)
        logger.info(
            f"SharedMLP prefetcher weights registered "
            f"(hidden K {hidden}→{k_pad_hidden}, gate/up N {inter_per_dev}→{n_pad}, "
            f"down K {inter_per_dev}→{k_pad_down}, down N {hidden}→{n_pad_down}, ring={ring})"
        )

    def _setup_dram_sharded(self, mesh_device, tp):
        """Precompute DRAM-sharded weight memory configs, matmul program configs and
        the L1 width-sharded activation config for the decode path.
        """
        dram_cores = mesh_device.dram_grid_size().x
        assert mesh_device.dram_grid_size().y == 1, "DRAM sharding assumes a single DRAM row"
        self._dram_cores = dram_cores

        hidden = self.hidden_size
        inter_per_dev = self.intermediate_size // tp
        assert self.intermediate_size % tp == 0, f"intermediate_size {self.intermediate_size} not divisible by tp {tp}"

        gu_k, gu_n = hidden, inter_per_dev
        dn_k, dn_n = inter_per_dev, hidden

        for name, k, n in (("gate/up", gu_k, gu_n), ("down", dn_k, dn_n)):
            assert k % TILE == 0 and n % TILE == 0, f"{name} dims ({k},{n}) not tile-aligned"

        gu_rows, gu_cols = _find_grid_k_n(gu_k // TILE, gu_n // TILE)
        dn_rows, dn_cols = _find_grid_k_n(dn_k // TILE, dn_n // TILE)
        self._gate_up_cores = gu_rows * gu_cols
        self._down_cores = dn_rows * dn_cols

        assert gu_k % (TILE * self._gate_up_cores) == 0, "gate/up K not divisible by tile*cores"
        assert dn_k % (TILE * self._down_cores) == 0, "down K not divisible by tile*cores"

        self._gate_up_weight_mem_config = self._dram_weight_mem_config(gu_k, gu_n, dram_cores)
        self._down_weight_mem_config = self._dram_weight_mem_config(dn_k, dn_n, dram_cores)

        m = TILE
        self._gate_up_in_mem_config = ttnn.create_sharded_memory_config(
            (m, gu_k // self._gate_up_cores),
            ttnn.CoreGrid(y=gu_rows, x=gu_cols),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self._down_in_mem_config = ttnn.create_sharded_memory_config(
            (m, dn_k // self._down_cores),
            ttnn.CoreGrid(y=dn_rows, x=dn_cols),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        gelu_approx = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)
        self._gate_prog_config = self._dram_matmul_prog_config(m, gu_k, gu_n, self._gate_up_cores, gelu_approx)
        self._up_prog_config = self._dram_matmul_prog_config(m, gu_k, gu_n, self._gate_up_cores, None)
        self._down_prog_config = self._dram_matmul_prog_config(m, dn_k, dn_n, self._down_cores, None)

    @staticmethod
    def _dram_weight_mem_config(k, n, dram_cores):
        padded_n = math.ceil(n / (TILE * dram_cores)) * (TILE * dram_cores)
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
        shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    @staticmethod
    def _dram_matmul_prog_config(m, k, n, num_cores, fused_activation):
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=_largest_divisor(k // (TILE * num_cores)),
            per_core_M=math.ceil(m / TILE),
            per_core_N=math.ceil(n / (TILE * num_cores)),
            fused_activation=fused_activation,
        )

    def __call__(self, hidden_states):
        """GeGLU MLP forward with TP support (gate/up column-parallel, down row-parallel + allreduce)."""
        if self.use_prefetcher and hidden_states.shape[-2] <= TILE:
            return self._forward_prefetcher(hidden_states)
        if self.use_dram_sharded and hidden_states.shape[-2] <= TILE:
            return self._forward_dram_sharded(hidden_states)

        gate = ttnn.linear(hidden_states, self.gate_proj, activation="gelu_approx")
        up = ttnn.linear(hidden_states, self.up_proj)

        hidden = ttnn.mul(gate, up)
        gate.deallocate(True)
        up.deallocate(True)

        output = ttnn.linear(hidden, self.down_proj)
        hidden.deallocate(True)

        if self.mesh_config is not None and self.mesh_config.tp > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)

        return output

    def _forward_prefetcher(self, hidden_states):
        """Decode GeGLU with Prefetcher ring-1D matmuls (global CB + worker sub-device)."""
        pf = self.prefetcher
        # pad/slice need a solid worker rectangle; fragmented receivers-only
        # grids can still trip "kernel group cores do not match sub device".
        worker_grids = pf.dynamic_worker_core_grid(16)

        # Pad hidden K so width shards across ring_size are tile-aligned.
        x = _pf_pad_width(hidden_states, self._pf_k_pad_hidden, worker_grids)
        x = ttnn.to_memory_config(x, self._pf_in_mem)

        gate = ttnn.linear(
            x,
            self._gate_proj_pf,
            program_config=self._pf_gate_pc,
            memory_config=self._pf_gate_out_mem,
            global_cb=pf.global_cb,
            sub_device_id=pf.worker_sub_device_id,
        )
        up = ttnn.linear(
            x,
            self._up_proj_pf,
            program_config=self._pf_up_pc,
            memory_config=self._pf_gate_out_mem,
            global_cb=pf.global_cb,
            sub_device_id=pf.worker_sub_device_id,
        )
        x.deallocate(True)

        # Ring matmul has no fused activation — apply approx GELU on the mul
        # (peer uses SILU the same way; Gemma4 GeGLU needs tanh-approx GELU).
        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)],
            memory_config=gate.memory_config(),
            sub_device_id=pf.worker_sub_device_id,
        )
        gate.deallocate(True)
        up.deallocate(True)

        # gate/up N and down K are both ring-padded to the same width (6144),
        # so the mul output is already a valid down activation — skip crop/re-pad
        # (avoids untilize on width-sharded H=1 tensors).
        output = ttnn.linear(
            hidden,
            self._down_proj_pf,
            program_config=self._pf_down_pc,
            memory_config=self._pf_down_out_mem,
            global_cb=pf.global_cb,
            sub_device_id=pf.worker_sub_device_id,
        )
        hidden.deallocate(True)

        output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)
        # Crop ring-padded down N back to logical hidden.
        cropped = _pf_slice_width(output, self.hidden_size, worker_grids)
        if cropped is not output:
            output.deallocate(True)
            output = cropped

        if self.mesh_config is not None and self.mesh_config.tp > 1:
            output = ccl_allreduce(
                output,
                self.mesh_config,
                self.ccl_manager,
                subdevice_id=pf.worker_sub_device_id,
            )
        return output

    def _forward_dram_sharded(self, hidden_states):
        """Decode GeGLU with DRAM-sharded (peak-BW) weight matmuls."""
        x = ttnn.to_memory_config(hidden_states, self._gate_up_in_mem_config)

        gate = ttnn.linear(
            x,
            self._gate_proj_ds,
            program_config=self._gate_prog_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        up = ttnn.linear(
            x,
            self._up_proj_ds,
            program_config=self._up_prog_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        x.deallocate(True)

        hidden = ttnn.mul(gate, up)
        gate.deallocate(True)
        up.deallocate(True)

        hidden = ttnn.to_memory_config(hidden, self._down_in_mem_config)

        output = ttnn.linear(
            hidden,
            self._down_proj_ds,
            program_config=self._down_prog_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        hidden.deallocate(True)

        output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)

        if self.mesh_config is not None and self.mesh_config.tp > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)

        return output
