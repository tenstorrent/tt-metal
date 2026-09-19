# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""One fused anti-aliased SnakeBeta activation: ``UpSample1d(2x) -> SnakeBeta -> DownSample1d(2x)`` as a single
``generic_op`` program (kernels in ``layers/kernels/aa_snake_*.cpp``) instead of the ~21-program chain of
``Activation1d``.

The rewrite is exact. With ``s = 2 t`` (the upsampler's scaled taps) and ``base[r] = x[clamp(r - 3)]``, the
upsampled signal's even and odd phases are unit-stride 6-tap filters, ``E[q] = sum_j s[2j] base[q + j]`` and
``O[q] = sum_j s[2j + 1] base[q + 1 + j]``; SnakeBeta applies per channel; the stride-2 12-tap downsampler is
``out[n] = sum_k t[k] z[clamp(2n + k - 5)]`` with ``z[2q] = E[q]``, ``z[2q + 1] = O[q]``. Each tap is an fp32
SFPU multiply then add in tap order, the arithmetic of the depthwise conv1d kernel today's chain runs, so the
result is bit-identical to ``Activation1d`` at ``t_pad = 0``.

Layout: fp32 row-major sticks; a 4 KB block of ``R = 1024 / C`` consecutive sticks is one fp32 "tile" to the
unpacker, which makes the elementwise math layout-blind and removes every tilize/untilize. Time-packed inputs
``(B, T / k, k C)`` need no special handling beyond their DRAM page size: the pack factor is read off the tensor
width. Under T-sharding the ``_t_neighbor_pad`` replicate halo (5 sticks per side, rounded up to packed rows)
supplies the neighbours; the sequence-end clamps run in-kernel on the first/last device.
"""

from __future__ import annotations

import struct

import torch

import ttnn

from ..parallel.config import ParallelFactor
from ..parallel.manager import CCLManager
from .audio_ops import _make_kaiser_sinc_kernel_1d, _t_neighbor_pad
from .module import Module, Parameter

KERNEL_DIR = "models/tt_dit/layers/kernels"
# Circular buffer indices shared with the kernels (compile-time args 0-7).
CB_X, CB_UP, CB_AB, CB_E, CB_O, CB_DN, CB_FL, CB_OUT = 0, 1, 2, 3, 4, 5, 6, 16
_TILE = 4096
_TAPS = 12
_DRAM_READ_ALIGN = 64  # Blackhole NoC DRAM read alignment (NOC_DRAM_READ_ALIGNMENT_BYTES)


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


class FusedActivation1d(Module):
    """Drop-in for ``Activation1d(channels, SnakeBeta(alpha_logscale=True))``: same state keys (``act.alpha``,
    ``act.beta``, optional ``upsample.filter`` / ``downsample.lowpass.filter``), same ``(B, T, C)`` or packed
    ``(B, T / k, k C)`` fp32 ROW_MAJOR input and output. ``stage`` is a bring-up knob: 0 copies the input through
    the whole gather machinery, 1 runs the resampler taps without the activation, 2 is the real thing.
    """

    def __init__(
        self,
        *,
        channels: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
        alpha_logscale: bool = True,
        stage: int = 2,
    ) -> None:
        super().__init__()
        if dtype != ttnn.float32:
            raise ValueError("FusedActivation1d is an fp32 kernel")
        if channels < 8 or 1024 % channels:
            raise ValueError(f"channels must divide 1024 and be >= 8, got {channels}")
        if parallel_config is not None and not isinstance(parallel_config, ParallelFactor):
            raise NotImplementedError("FusedActivation1d supports single-axis T-sharding only")
        self.channels = channels
        self.rows_per_tile = 1024 // channels
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.alpha_logscale = alpha_logscale
        self.stage = int(stage)
        self.eps = 1e-9
        self._sharded = parallel_config is not None and parallel_config.factor > 1
        if self._sharded:
            assert ccl_manager is not None, "T-sharding requires ccl_manager"
            assert tuple(mesh_device.shape)[parallel_config.mesh_axis] == parallel_config.factor
        taps = _make_kaiser_sinc_kernel_1d(cutoff=0.25, half_width=0.3, kernel_size=_TAPS).tolist()
        self._up_taps = list(taps)
        self._down_taps = list(taps)
        # Alpha (row 0) and beta (row 1), each the per-channel vector repeated R times: one fp32 tile each.
        self.ab = Parameter(
            total_shape=[1, 2, 1024], device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32
        )
        self._flags: ttnn.Tensor | None = None
        self._programs: dict = {}

    # -- state -------------------------------------------------------------------------------------------------

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "act.alpha" in state and "act.beta" in state:
            alpha = state.pop("act.alpha")
            beta = state.pop("act.beta")
            # The same steps as SnakeBeta._prepare_torch_state so the values match bit for bit.
            if self.alpha_logscale:
                alpha = torch.exp(alpha)
                beta = torch.exp(beta)
            alpha = alpha.reshape(-1)
            beta = beta.reshape(-1) + self.eps
            assert alpha.numel() == self.channels and beta.numel() == self.channels
            r = self.rows_per_tile
            state["ab"] = torch.stack([alpha.float().repeat(r), beta.float().repeat(r)]).reshape(1, 2, 1024).contiguous()
        if "upsample.filter" in state:
            self._up_taps = state.pop("upsample.filter").reshape(-1).float().tolist()
        if "downsample.lowpass.filter" in state:
            self._down_taps = state.pop("downsample.lowpass.filter").reshape(-1).float().tolist()
        assert len(self._up_taps) == _TAPS and len(self._down_taps) == _TAPS

    def _flags_tensor(self) -> ttnn.Tensor:
        """Per-device ``[is_first, is_last]`` along the time axis (uint32, one 64 B page)."""
        if self._flags is None:
            if self._sharded:
                factor, axis = self.parallel_config.factor, self.parallel_config.mesh_axis
                flags = torch.zeros(factor, 1, 16, dtype=torch.int32)
                flags[0, 0, 0] = 1
                flags[factor - 1, 0, 1] = 1
                dims = [None, None]
                dims[axis] = 0
                mapper = ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims)
            else:
                flags = torch.zeros(1, 1, 16, dtype=torch.int32)
                flags[0, 0, 0] = 1
                flags[0, 0, 1] = 1
                mapper = None
            self._flags = ttnn.from_torch(
                flags,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
        return self._flags

    # -- program -----------------------------------------------------------------------------------------------

    def _build(self, batch: int, t_sticks: int, pack: int, halo: int, x: ttnn.Tensor, out: ttnn.Tensor) -> dict:
        C, R = self.channels, self.rows_per_tile
        stick = C * 4
        page = pack * stick
        grid = self.mesh_device.compute_with_storage_grid_size()
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        cores = [(cx, cy) for cy in range(grid.y) for cx in range(grid.x)]
        per_batch = len(cores) // batch
        assert per_batch >= 1, f"batch {batch} exceeds the {len(cores)}-core grid"
        tiles = -(-t_sticks // R)
        # (b, first output stick, output tiles) per core; the spare cores get no work and return at once.
        work = []
        for b in range(batch):
            base, rem = divmod(tiles, per_batch)
            o0 = 0
            for j in range(per_batch):
                n = base + (1 if j < rem else 0)
                work.append((b, o0, n))
                o0 += n * R
        work += [(0, 0, 0)] * (len(cores) - len(work))
        max_tiles = max(n for _, _, n in work)
        nb_extra = -(-6 // R)  # E/O blocks cover the output range +-3 sticks
        max_blocks = max_tiles + nb_extra

        rt = ttnn.RuntimeArgs()
        for (cx, cy), (b, o0, n) in zip(cores, work, strict=True):
            rt[cx][cy] = [b, o0, n]

        ct = [
            CB_X, CB_UP, CB_AB, CB_E, CB_O, CB_DN, CB_OUT, CB_FL,
            C, R, pack, t_sticks, halo,
            (t_sticks + 2 * halo) // pack,  # input pages per batch item
            t_sticks // pack,  # output pages per batch item
            self.stage, 0, nb_extra,
        ]  # fmt: skip
        reader_ct = list(ct)
        for t in (x, self.ab.data, self._flags_tensor()):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = list(ct) + ttnn.TensorAccessorArgs(out).get_compile_time_args()

        def cb(index, page_bytes, pages, fmt=ttnn.float32):
            fmt_desc = ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page_bytes)
            return ttnn.CBDescriptor(total_size=page_bytes * pages, core_ranges=core_grid, format_descriptors=[fmt_desc])

        def align64(n):
            return -(-n // _DRAM_READ_ALIGN) * _DRAM_READ_ALIGN

        # Staging holds the block range +-3 sticks plus a page of slack on each side (see the reader); every CB
        # size is a multiple of 64 B so the staging base keeps the DRAM read alignment.
        stage_bytes = align64((max_blocks * R + 6) * stick + 2 * page)
        cbs = [
            cb(CB_X, stage_bytes, 1),
            cb(CB_UP, _TILE, 14),
            cb(CB_AB, _TILE, 2),
            cb(CB_E, _TILE, max_blocks),
            cb(CB_O, _TILE, max_blocks),
            cb(CB_DN, _TILE, 24),
            cb(CB_FL, 64, 1, ttnn.int32),
            cb(CB_OUT, _TILE, 2),
        ]

        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64  # one entry per circular buffer slot
        for i in (CB_UP, CB_AB, CB_DN):
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        vec = getattr(ttnn, "VectorUnpackToDestMode", None)
        compute_cfg.unpack_to_dest_mode = vec(modes) if vec is not None else modes

        taps = [_f32_bits(2.0 * t) for t in self._up_taps] + [_f32_bits(t) for t in self._down_taps]
        return dict(
            core_grid=core_grid,
            rt=rt,
            reader_ct=reader_ct,
            writer_ct=writer_ct,
            compute_ct=list(ct),
            cbs=cbs,
            compute_cfg=compute_cfg,
            taps=taps,
            # Deterministic across processes (no str hashing): the shape key is what makes the program distinct.
            hash=(0x5A5 << 52)
            | (C << 40)
            | (pack << 36)
            | (batch << 32)
            | (t_sticks << 12)
            | (halo << 6)
            | (self.stage << 4)
            | ((grid.x * grid.y) & 0xF),
        )

    def _descriptor(self, batch, t_sticks, pack, halo, x, out) -> ttnn.ProgramDescriptor:
        key = (batch, t_sticks, pack, halo)
        built = self._programs.get(key)
        if built is None:
            built = self._build(batch, t_sticks, pack, halo, x, out)
            self._programs[key] = built
        common = dict(core_ranges=built["core_grid"], runtime_args=built["rt"])
        reader = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/aa_snake_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            compile_time_args=built["reader_ct"],
            common_runtime_args=[x.buffer_address(), self.ab.data.buffer_address(), self._flags_tensor().buffer_address()],
            config=ttnn.ReaderConfigDescriptor(),
            **common,
        )
        writer = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/aa_snake_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            compile_time_args=built["writer_ct"],
            common_runtime_args=[out.buffer_address()],
            config=ttnn.WriterConfigDescriptor(),
            **common,
        )
        compute = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_DIR}/aa_snake_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            compile_time_args=built["compute_ct"],
            common_runtime_args=list(built["taps"]),
            config=built["compute_cfg"],
            **common,
        )
        program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=built["cbs"])
        try:
            program.custom_program_hash = built["hash"]
        except (AttributeError, TypeError):
            pass
        return program

    # -- forward -----------------------------------------------------------------------------------------------

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        assert x_BTC.layout == ttnn.ROW_MAJOR_LAYOUT and x_BTC.dtype == ttnn.float32
        batch, rows, width = (int(d) for d in x_BTC.shape)
        C = self.channels
        assert width % C == 0, f"row width {width} is not a multiple of {C} channels"
        pack = width // C
        # DRAM reads need 64 B alignment: rows narrower than that (C = 8 unpacked) run two per row.
        x = x_BTC
        unpacked_shape = None
        if width * 4 < _DRAM_READ_ALIGN:
            k2 = _DRAM_READ_ALIGN // (width * 4)
            assert rows % k2 == 0, f"{rows} rows cannot be packed {k2} per row"
            x = ttnn.reshape(x_BTC, (batch, rows // k2, width * k2))
            unpacked_shape = (batch, rows, width)
            rows, width, pack = rows // k2, width * k2, pack * k2
        t_sticks = rows * pack
        assert x.buffer_aligned_page_size() == width * 4, "the kernel reads whole unpadded DRAM pages"
        # The accessor args baked into the kernels assume DRAM-interleaved pages, and the program hash packs
        # (t_sticks, batch, pack) into fixed bit fields.
        assert x.memory_config().buffer_type == ttnn.BufferType.DRAM and not x.is_sharded(), "x must be DRAM interleaved"
        assert t_sticks < (1 << 20) and batch < 16 and pack < 16, f"shape ({batch}, {t_sticks}, pack {pack}) exceeds the key's fields"

        if self._sharded:
            pad_rows = -(-5 // pack)
            halo_t = _t_neighbor_pad(
                x,
                pad_left=pad_rows,
                pad_right=pad_rows,
                parallel_config=self.parallel_config,
                ccl_manager=self.ccl_manager,
                padding_mode="replicate",
            )
            halo = pad_rows * pack
        else:
            halo_t, halo = x, 0

        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([batch, rows, width]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        program = self._descriptor(batch, t_sticks, pack, halo, halo_t, out)
        out = ttnn.generic_op([halo_t, self.ab.data, self._flags_tensor(), out], program)
        if x is not x_BTC:
            ttnn.deallocate(x)
        if unpacked_shape is not None:
            packed = out
            out = ttnn.reshape(packed, unpacked_shape)
            if out.buffer_address() != packed.buffer_address():
                ttnn.deallocate(packed)
        return out
