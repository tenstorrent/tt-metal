# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in joint GDN primary/correction projection candidate.

No TTNN import at module load. All weight reads/repacking are setup-only. Runtime methods receive the current
resource owner explicitly; class dispatch covers both prefill and decode.
"""

import math

KEY = "packed_gdn_correction_joint"
PRIMARY_N = 4160
CORRECTION_N = 2624
JOINT_N = PRIMARY_N + CORRECTION_N


def geometry(*, banks=8, readers=2, cores=8, block=2):
    if min(banks, readers, cores, block) <= 0 or 160 % cores or 160 % block:
        raise ValueError("Geometry must divide K=5120 into 32-channel units")
    quantum = 32 * math.lcm(banks * readers, cores)
    padded = lambda n: math.ceil(n / quantum) * quantum
    return {
        "banks": banks,
        "readers": readers,
        "cores": cores,
        "block": block,
        "primary_n": PRIMARY_N,
        "correction_n": CORRECTION_N,
        "joint_n": JOINT_N,
        "primary_padded_n": padded(PRIMARY_N),
        "correction_padded_n": padded(CORRECTION_N),
        "joint_padded_n": padded(JOINT_N),
        "joint_bf8_bytes_per_device": 160 * (padded(JOINT_N) // 32) * 1088,
    }


def precision_contract(output_dtype="float32", fp32_dest_acc_en=True):
    """Independent output-pack and matmul-accumulation choices; no device imports."""
    if output_dtype not in ("float32", "bfloat16") or type(fp32_dest_acc_en) is not bool:
        raise ValueError("Joint precision requires float32/bfloat16 output and a boolean FP32-DST flag")
    if output_dtype == "float32" and not fp32_dest_acc_en:
        raise ValueError("Reduced joint accumulation requires explicit bfloat16 output")
    return {"output_dtype": output_dtype, "fp32_dest_acc_en": fp32_dest_acc_en}


def combine_storage(output_dtype):
    precision_contract(output_dtype)
    page = 4096 if output_dtype == "float32" else 2048
    return (
        (0, page * 2, page, output_dtype),
        (1, page * 2, page, output_dtype),
        (2, 4096, 2048, "bfloat16"),
        (3, page, page, output_dtype),
    )


def combine_reader(reader, output_dtype):
    """Only the five audited input-page/zero-byte literals differ for BF16."""
    precision_contract(output_dtype)
    if output_dtype == "float32":
        return reader
    if reader.count("4096") != 5:
        raise AssertionError("Combine reader page literals changed; re-audit the BF16 adapter")
    return reader.replace("4096", "2048")


def prefill_plan(rows, physical_n, policy, options=None, *, fp32_dest_acc_en=True):
    """CPU-only 2D geometry; FP32 defaults retain the inherited four-tile plan."""
    options = options or {}
    known = {"grid", "in0_block_w", "subblock_h", "subblock_w", "input_l1", "out_block_h", "row_cap"}
    if set(options) - known:
        raise ValueError(f"Unknown joint prefill options: {set(options) - known}")
    inherited = policy.get("prefill_projection_configs", {}).get("packed_gdn", {})
    options = {**inherited, **options}
    gx, gy = options.get("grid", policy.get("prefill_grid", (11, 10)))
    if min(rows, physical_n, gx, gy) < 1 or physical_n % 32:
        raise ValueError("Invalid joint prefill dimensions")
    pm = math.ceil(rows / 32 / gy)
    pn = math.ceil(physical_n / 32 / gx)
    if type(fp32_dest_acc_en) is not bool:
        raise ValueError("FP32-DST flag must be boolean")
    widths = (1, 2, 4) if fp32_dest_acc_en else (1, 2, 4, 8)
    limit = 4 if fp32_dest_acc_en else 8
    sw = options.get("subblock_w", max(v for v in widths if pn % v == 0))
    requested_bh = options.get("out_block_h", 1 if policy.get("prefill_height_block", False) else pm)
    if requested_bh < 1:
        raise ValueError("out_block_h must be positive")
    bh = max(v for v in range(1, pm + 1) if v <= requested_bh and pm % v == 0)
    requested_h = min(bh, options.get("subblock_h", 4 if policy.get("prefill_subblock_h", False) else 1))
    if sw not in widths or requested_h < 1:
        raise ValueError("Invalid joint prefill subblock")
    sh = max(v for v in widths if v <= requested_h and bh % v == 0 and v * sw <= limit)
    kb = options.get("in0_block_w", policy.get("prefill_k", 8))
    if not isinstance(kb, int) or kb <= 0 or 160 % kb or pn % sw:
        raise ValueError("Joint prefill K/N geometry is not divisible")
    return {
        "compute_with_storage_grid_size": (gx, gy),
        "in0_block_w": kb,
        "out_subblock_h": sh,
        "out_subblock_w": sw,
        "per_core_M": pm,
        "per_core_N": pn,
        "out_block_h": bh,
        "out_block_w": pn,
        "transpose_mcast": False,
        "fused_activation": None,
        "fuse_batch": True,
    }


def combine_window_offset(packed_shape, output_shape, row_offset):
    """Bounded flat tile-row window; returns the destination tile offset."""
    if type(row_offset) is not int or row_offset < 0 or row_offset % 32:
        raise ValueError("Combine window offset must be a nonnegative multiple of 32")
    if (
        len(packed_shape) != 3
        or len(output_shape) != 3
        or packed_shape[0] != 1
        or output_shape[0] != 1
        or packed_shape[1] < 32
        or packed_shape[1] % 32
        or packed_shape[2] < JOINT_N
        or packed_shape[2] % 32
        or output_shape[1] < 32
        or output_shape[1] % 32
        or output_shape[2] != PRIMARY_N
    ):
        raise ValueError("Combine window requires aligned flat joint input and compact output")
    if row_offset + packed_shape[1] > output_shape[1]:
        raise ValueError("Combine window exceeds destination row capacity")
    if output_shape[1] // 32 * (PRIMARY_N // 32) >= 2**32:
        raise ValueError("Combine window exceeds uint32 tile addressing")
    return row_offset // 32 * (PRIMARY_N // 32)


def combine_window_writer(writer):
    """Setup-only source adaptation; input/compute indices remain local."""
    anchor = "const uint32_t start=get_arg_val<uint32_t>(1),count=get_arg_val<uint32_t>(2);"
    if writer.count(anchor) != 1 or writer.count("{.page_id=tile}") != 1:
        raise AssertionError("Combine writer changed; re-audit destination window")
    return writer.replace(anchor, anchor + "\n    const uint32_t output_tile_offset=get_arg_val<uint32_t>(3);").replace(
        "{.page_id=tile}", "{.page_id=tile+output_tile_offset}"
    )


class PackedCorrection:
    def __init__(
        self,
        layer,
        *,
        cores=None,
        readers=None,
        block=None,
        combine="fused",
        prefill_direct_output=False,
        prefill_output_dtype="float32",
        decode_output_dtype="float32",
        prefill_fp32_dest_acc_en=True,
        decode_fp32_dest_acc_en=True,
    ):
        self.precision = {
            "prefill": precision_contract(prefill_output_dtype, prefill_fp32_dest_acc_en),
            "decode": precision_contract(decode_output_dtype, decode_fp32_dest_acc_en),
        }
        if combine == "slice" and any(v["output_dtype"] != "float32" for v in self.precision.values()):
            raise ValueError("BF16 joint output requires the direct fused combine")
        if type(prefill_direct_output) is not bool:
            raise ValueError("prefill_direct_output must be boolean")
        if prefill_direct_output and (combine != "fused" or prefill_output_dtype != "float32"):
            raise ValueError("Direct prefill output requires fused combine and FP32 joint scratch")
        self.prefill_direct_output = prefill_direct_output
        import torch
        import ttnn
        from . import multichip_decoder as local

        if layer.kind != "linear_attention" or not layer.narrow_gdn_correction:
            raise ValueError("This candidate requires local narrow GDN correction")
        if not layer.policy.get("dram_sharded") or not layer.policy.get("flat_decode"):
            raise ValueError("This candidate requires DRAM-sharded compact decode")
        if combine not in ("fused", "slice"):
            raise ValueError("combine must be fused or slice")
        keys = ("packed_gdn", layer.weight_corrections["packed_gdn"])
        if any(layer.weights[k].dtype != ttnn.bfloat8_b for k in keys):
            raise ValueError("One packed matmul requires both original matrices to be BFP8")
        if layer.policy.get("attention_activation_dtype", "bfloat16") != "bfloat16":
            raise ValueError("This first family comparison preserves the BF16 activation contract")
        self.combine_mode = combine
        self.prefill_config = {}
        base = {**layer.policy, **layer.policy.get("geometry", {}).get("packed_gdn", {})}
        grid = layer.mesh.dram_grid_size()
        self.geometry = geometry(
            banks=grid.x * grid.y,
            cores=cores or base["cores"],
            readers=readers or base["readers"],
            block=block or base["block"],
        )
        g = self.geometry
        # Copy decoded quantized values, not original HF floats. Both column
        # boundaries are tile-aligned, preserving each BFP8 exponent group.
        original = [
            [ttnn.to_torch(t).float()[:, :n].contiguous() for t in ttnn.get_device_tensors(layer.weights[k])]
            for k, n in zip(keys, (PRIMARY_N, CORRECTION_N))
        ]
        if any(len(shards) != 4 for shards in original):
            raise ValueError("The candidate requires actual TP4 weights")
        values = [torch.cat(pair, -1) for pair in zip(*original)]
        values = [torch.nn.functional.pad(v, (0, g["joint_padded_n"] - JOINT_N)) for v in values]
        packed = layer._upload_shards(values, ttnn.bfloat8_b)
        readback = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(packed)]
        if not all(torch.equal(expected, actual) for expected, actual in zip(values, readback)):
            raise AssertionError("Joint upload changed original quantized matrix values")
        dram = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram, (5120, g["joint_padded_n"] // g["banks"]), ttnn.ShardOrientation.ROW_MAJOR),
        )
        layer.weights[KEY] = ttnn.to_memory_config(packed, memory)
        workers = ttnn.num_cores_to_corerangeset(g["cores"], layer.mesh.compute_with_storage_grid_size(), row_wise=True)
        self.input_mem = ttnn.create_sharded_memory_config(
            (32, 5120 // g["cores"]), workers, ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
        )
        self.program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=g["block"],
            per_core_M=1,
            per_core_N=g["joint_padded_n"] // 32 // g["cores"],
            num_workers_per_dram_bank=g["readers"],
            use_multicore_pipeline=base["pipeline"],
        )
        # Raw matrix width is6784; the corrected public projection width is4160.
        # Metadata/configs are immutable after setup and contain no layer owner.
        layer.projection_info[KEY] = {
            "input_mem": self.input_mem,
            "program": self.program,
            "logical_n": JOINT_N,
            "corrected_output_n": PRIMARY_N,
            "role": "attention",
            "geometry": dict(g),
        }
        before = "{.page_id=row*rn+rc}"
        if local._LOCAL_NARROW_READER.count(before) != 1:
            raise AssertionError("Combine reader source changed; audit the packed offset adapter")
        self.reader = local._LOCAL_NARROW_READER.replace(before, "{.page_id=row*rn+130+rc}")
        self.bf16_reader = combine_reader(self.reader, "bfloat16")
        self.writer = local._NARROW_GDN_COMBINE_WRITER
        self.window_writer = combine_window_writer(self.writer) if prefill_direct_output else None
        self.compute = local._LOCAL_NARROW_COMPUTE
        layer.policy["packed_correction_candidate"] = {
            **g,
            "combine": combine,
            "scope": "decode_only",
            "setup_quantized_values_exact": True,
            "original_prefill_weights_retained": True,
        }
        if prefill_direct_output:
            layer.policy["packed_correction_candidate"]["prefill_direct_output"] = True
        # No default policy metadata/config changes; opt-in provenance is explicit.
        if any(v != precision_contract() for v in self.precision.values()):
            layer.policy["packed_correction_candidate"]["phase_precision"] = dict(self.precision)

    def _phase_config(self, owner, phase):
        import ttnn

        contract = self.precision[phase]
        config = owner.matmul_configs["attention"]
        if not contract["fp32_dest_acc_en"]:
            config = ttnn.init_device_compute_kernel_config(
                owner.mesh.arch(),
                math_fidelity=config.math_fidelity,
                math_approx_mode=config.math_approx_mode,
                fp32_dest_acc_en=False,
                packer_l1_acc=config.packer_l1_acc,
                dst_full_sync_en=False,
                throttle_level=config.throttle_level,
            )
        return getattr(ttnn, contract["output_dtype"]), config

    def project(self, owner, x):
        return self._project_compact(owner, x, phase="decode")

    def _project_compact(self, owner, x, *, phase):
        import ttnn

        if len(x.shape) != 3 or x.shape[0] != 1 or not 1 <= x.shape[1] <= 32 or x.shape[2] != 5120:
            raise ValueError(f"Unexpected compact decode projection shape {x.shape}")
        if x.dtype != ttnn.bfloat16:
            raise ValueError("The joint candidate preserves BF16 projection inputs")
        dtype, config = self._phase_config(owner, phase)
        output = ttnn.linear(
            ttnn.to_memory_config(x, self.input_mem),
            owner.weights[KEY],
            program_config=self.program,
            compute_kernel_config=config,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
            dtype=dtype,
        )
        return self._finish_projection(owner, output)

    def project_prefill(self, owner, x):
        """Joint projection for any logical [B,S,5120], without state changes.

        Cap/pad only the token-independent matmul. The caller retains ownership
        of temporal GDN chunking and its live FP32 recurrent state.
        """
        import ttnn

        if len(x.shape) != 3 or x.shape[2] != 5120 or min(tuple(x.shape)[:2]) < 1:
            raise ValueError(f"Unexpected logical prefill projection shape {x.shape}")
        if x.dtype != ttnn.bfloat16:
            raise ValueError("Joint prefill preserves BF16 projection inputs")
        batch, seq, _ = tuple(x.shape)
        rows = batch * seq
        flat = ttnn.reshape(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), (1, rows, 5120))
        if rows <= 32:
            return ttnn.reshape(self._project_compact(owner, flat, phase="prefill"), (batch, seq, PRIMARY_N))
        dtype, compute_config = self._phase_config(owner, "prefill")
        options = self.prefill_config
        cap = options.get("row_cap", owner.policy.get("matmul_row_cap", 512))
        if not isinstance(cap, int) or cap < 32 or cap % 32:
            raise ValueError("Joint prefill row cap must be a positive multiple of 32")
        padded_rows = (rows + 31) // 32 * 32
        if rows != padded_rows:
            flat = ttnn.pad(flat, ((0, 0), (0, padded_rows - rows), (0, 0)), value=0.0)
        # One invocation-owned destination, no length-keyed persistent cache.
        # Single-piece prefill retains its original allocation/program path.
        direct_output = self.prefill_direct_output and padded_rows > cap
        output = None
        if direct_output:
            output = ttnn.empty(
                (1, padded_rows, PRIMARY_N),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=flat.device(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        parts = []
        for start in range(0, padded_rows, cap):
            stop = min(start + cap, padded_rows)
            chunk = flat if start == 0 and stop == padded_rows else flat[:, start:stop, :]
            plan = prefill_plan(
                stop - start,
                self.geometry["joint_padded_n"],
                owner.policy,
                options,
                fp32_dest_acc_en=self.precision["prefill"]["fp32_dest_acc_en"],
            )
            config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**plan)
            if options.get("input_l1", owner.policy.get("prefill_input_l1", False)):
                chunk = ttnn.to_memory_config(chunk, ttnn.L1_MEMORY_CONFIG)
            projected = ttnn.linear(
                chunk,
                owner.weights[KEY],
                program_config=config,
                compute_kernel_config=compute_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=dtype,
            )
            if direct_output:
                self._combine(projected, output=output, row_offset=start)
            else:
                parts.append(self._finish_projection(owner, projected))
        if not direct_output:
            output = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=1)
        if rows != padded_rows:
            output = output[:, :rows, :]
        return ttnn.reshape(output, (batch, seq, PRIMARY_N))

    def _finish_projection(self, owner, output):
        import ttnn

        if self.combine_mode == "slice":
            output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
            return owner._combine_correction(output[:, :, :PRIMARY_N], output[:, :, PRIMARY_N:JOINT_N], "packed_gdn")
        return self._combine(output)

    def _combine(self, packed, *, output=None, row_offset=0):
        """FP32 add/RNE-BF16 kernel, reading the joint dtype without a restore."""
        import ttnn

        assert packed.dtype in (ttnn.float32, ttnn.bfloat16) and packed.shape[-1] >= JOINT_N
        source_dtype = "float32" if packed.dtype == ttnn.float32 else "bfloat16"
        batch, seq, _ = tuple(packed.shape)
        window = output is not None
        output_tile_offset = 0
        if window:
            if not self.prefill_direct_output or self.window_writer is None:
                raise ValueError("Combine destination windows require setup opt-in")
            if (
                packed.dtype != ttnn.float32
                or packed.layout != ttnn.TILE_LAYOUT
                or packed.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                or tuple(packed.shape) != tuple(packed.padded_shape)
            ):
                raise ValueError("Combine window input must be aligned FP32 tile DRAM interleaved")
            if (
                output.dtype != ttnn.bfloat16
                or output.layout != ttnn.TILE_LAYOUT
                or output.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                or tuple(output.shape) != tuple(output.padded_shape)
                or output.device() != packed.device()
                or output.buffer_address() == packed.buffer_address()
            ):
                raise ValueError("Combine window output must be distinct aligned BF16 tile DRAM on the same device")
            output_tile_offset = combine_window_offset(tuple(packed.shape), tuple(output.shape), row_offset)
        else:
            if row_offset != 0:
                raise ValueError("Combine row offset requires a supplied destination")
            output = ttnn.empty(
                (batch, seq, PRIMARY_N),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=packed.device(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        total = batch * (packed.padded_shape[-2] // 32) * 130
        grid = packed.device().compute_with_storage_grid_size()
        workers = min(total, grid.x * grid.y)
        coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
        rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
        start = 0
        for i, c in enumerate(coords):
            count = total // workers + (i < total % workers)
            # Both accessor bases point at one allocation; the reader applies
            # 130 tiles to the correction columns and uses one padded stride.
            rr[c.x][c.y] = [
                packed.buffer_address(),
                packed.buffer_address(),
                packed.padded_shape[-1] // 32,
                packed.padded_shape[-1] // 32,
                start,
                count,
            ]
            wr[c.x][c.y] = [output.buffer_address(), start, count]
            if window:
                wr[c.x][c.y].append(output_tile_offset)
            cr[c.x][c.y] = [start, count]
            start += count
        rct = list(ttnn.TensorAccessorArgs(packed).get_compile_time_args()) * 2

        def kernel(source, config, runtime, compile_time):
            return ttnn.KernelDescriptor(
                kernel_source=source,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=compile_time,
                runtime_args=runtime,
                config=config,
            )

        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        if source_dtype == "float32":
            for i in (0, 1, 3):
                modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
        kernels = [
            kernel(
                self.reader if source_dtype == "float32" else self.bf16_reader, ttnn.ReaderConfigDescriptor(), rr, rct
            ),
            kernel(
                self.window_writer if window else self.writer,
                ttnn.WriterConfigDescriptor(),
                wr,
                ttnn.TensorAccessorArgs(output).get_compile_time_args(),
            ),
            kernel(self.compute, config, cr, []),
        ]
        cbs = [
            ttnn.CBDescriptor(
                total_size=size,
                core_ranges=ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=i,
                        data_format=getattr(ttnn, data_format),
                        page_size=page,
                    )
                ],
            )
            for i, size, page, data_format in combine_storage(source_dtype)
        ]
        ttnn.generic_op([packed, output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return output


def install(layer, *, prefill=True, release_separate=True, prefill_config=None, **options):
    """Setup only. Class methods dispatch with the current resource owner.

    This production adapter requires both prefill and decode. No bound method,
    closure, shallow decoder view or runtime phase flag is installed.
    """
    if getattr(layer, "_packed_correction_candidate", None) is not None:
        raise ValueError("The packed correction candidate is already installed")
    if not prefill:
        raise ValueError("Owner-safe integration requires joint prefill and decode")
    if not getattr(type(layer), "_owner_safe_joint_gdn", False):
        raise ValueError("Owner-safe decoder class dispatch must be wired first")
    config = {
        **layer.policy.get("prefill_projection_configs", {}).get("packed_gdn", {}),
        **(prefill_config or {}),
    }
    base = {**layer.policy, **layer.policy.get("geometry", {}).get("packed_gdn", {})}
    grid = layer.mesh.dram_grid_size()
    g = geometry(
        banks=grid.x * grid.y,
        cores=options.get("cores") or base["cores"],
        readers=options.get("readers") or base["readers"],
        block=options.get("block") or base["block"],
    )
    prefill_plan(
        512, g["joint_padded_n"], layer.policy, config, fp32_dest_acc_en=options.get("prefill_fp32_dest_acc_en", True)
    )
    cap = config.get("row_cap", layer.policy.get("matmul_row_cap", 512))
    if not isinstance(cap, int) or cap < 32 or cap % 32:
        raise ValueError("Joint prefill row cap must be a positive multiple of32")
    candidate = PackedCorrection(layer, **options)
    candidate.prefill_config = config
    layer._packed_correction_candidate = candidate
    layer.policy["packed_correction_candidate"].update(
        scope="decode_and_prefill",
        prefill_config=dict(config),
        dispatch="class_methods_explicit_owner",
        corrected_output_n=PRIMARY_N,
    )
    if release_separate:
        correction_key = layer.weight_corrections.pop("packed_gdn")
        for key in ("packed_gdn", correction_key):
            layer.weights.pop(key)
            layer.projection_info.pop(key, None)
        layer.policy["packed_correction_candidate"]["original_prefill_weights_retained"] = False
    return layer


def cpu_check():
    import json
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(737)
    geometries = [geometry(cores=c, readers=r, block=2) for c in (4, 8, 16) for r in (1, 2, 3)]
    checks = []
    for batch in (1, 2, 3, 17, 32):
        # Small integer operands make the independent and packed CPU GEMMs
        # exactly representable, testing mapping without a rounding tolerance.
        x = torch.randint(-2, 3, (batch, 64)).float()
        p, r = (torch.randint(-2, 3, (64, n)).float() for n in (PRIMARY_N, CORRECTION_N))
        joint = x @ torch.cat((p, r), -1)
        primary, residual = x @ p, x @ r
        assert torch.equal(joint[:, :PRIMARY_N], primary)
        assert torch.equal(joint[:, PRIMARY_N:JOINT_N], residual)
        reference = primary.clone()
        reference[:, :2560] += residual[:, :2560]
        reference[:, 2560:4096] += 0.0
        reference[:, 4096:4160] += residual[:, 2560:2624]
        for col in range(130):
            start = col * 32
            actual = joint[:, start : start + 32]
            if col < 80 or col >= 128:
                rc = col if col < 80 else col - 48
                actual = actual + joint[:, (130 + rc) * 32 : (131 + rc) * 32]
            else:
                actual = actual + 0.0
            assert torch.equal(actual, reference[:, start : start + 32])
        checks.append(batch)
    prefill_checks = []
    for batch in (1, 2, 3, 17, 32):
        for seq in (1, 2, 31, 33, 513):
            for cap in (32, 128, 512, 1024):
                rows = batch * seq
                flat = torch.arange(rows).reshape(1, rows, 1)
                padded_rows = (rows + 31) // 32 * 32
                padded = torch.nn.functional.pad(flat, (0, 0, 0, padded_rows - rows), value=-1)
                parts = [padded[:, start : min(start + cap, padded_rows)] for start in range(0, padded_rows, cap)]
                output = torch.cat(parts, 1)[:, :rows].reshape(batch, seq, 1)
                assert torch.equal(output, torch.arange(rows).reshape(batch, seq, 1))
                prefill_checks.append({"batch": batch, "seq": seq, "row_cap": cap})
    plans = [prefill_plan(rows, 7168, {"prefill_subblock_h": True}) for rows in (32, 64, 128, 512, 1024)]
    result = {
        "pass": True,
        "kind": "cpu_mapping_only",
        "batches": checks,
        "geometries": geometries,
        "prefill_row_order_cases": len(prefill_checks),
        "prefill_example_plans": plans,
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    import hashlib
    import sys
    from pathlib import Path

    if sys.argv[1:] == ["--cpu-check"]:
        cpu_check()
        return
    if "--baseline" in sys.argv:
        raise ValueError("The candidate CLI measures TP4 only")
    from models.autoports.qwen_qwen3_6_27b.tests import run_multichip_decoder as runner

    class CandidateDecoder(runner.MultichipDecoder):
        @classmethod
        def from_state_dict(cls, *args, **kwargs):
            policy = dict(kwargs.get("policy") or {})
            policy.setdefault("packed_correction_options", {"prefill": True, "release_separate": True})
            kwargs["policy"] = policy
            layer = super().from_state_dict(*args, **kwargs)
            layer.policy["packed_correction_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            return layer

    runner.MultichipDecoder = CandidateDecoder
    runner.main()


if __name__ == "__main__":
    main()
