"""Stage-owned candidate graphs. Never imported by delivered runtime."""

import math

import ttnn

from ..tt.optimized_decoder import OptimizedDecoder


class PrefillCandidate(OptimizedDecoder):
    experiment = "prefill_1024_2d_8_8_4"

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        parts = cls.experiment.split("_")
        obj.chunk_size = int(parts[1])
        obj.prefill_kernel = parts[2]
        obj.prefill_grid = (int(parts[3]), int(parts[4]))
        obj.prefill_block = int(parts[5])
        obj.prefill_mblock = int(parts[6]) if len(parts) > 6 else 4
        obj.prefill_nblock = int(parts[7]) if len(parts) > 7 else 8
        return obj

    def _linear(self, x, w):
        if x.shape[2] <= 32 or self.prefill_kernel == "default":
            return super()._linear(x, w)
        m, k, n = x.padded_shape[-2], x.padded_shape[-1], w.padded_shape[-1]
        gx, gy = self.prefill_grid
        if self.prefill_kernel == "minimal":
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=self.prefill_mblock,
                K_block_size=self.prefill_block,
                N_block_size=self.prefill_nblock,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            )
            return ttnn.experimental.minimal_matmul(
                x, w, config=cfg, compute_kernel_config=self._weight_compute(w), dtype=ttnn.bfloat16
            )
        pm = math.ceil(m / 32 / gy)
        pn = math.ceil(n / 32 / gx)
        sw = next(s for s in [4, 3, 2, 1] if pn % s == 0)
        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=self.prefill_block,
            out_subblock_h=1,
            out_subblock_w=sw,
            per_core_M=pm,
            per_core_N=pn,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        return ttnn.linear(
            x,
            w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._weight_compute(w),
            program_config=cfg,
        )


class NormCandidate(OptimizedDecoder):
    experiment = "norm_4"

    def _norm(self, x):
        if x.shape[2] > 32:
            return super()._norm(x)
        shape = x.shape
        grouped = ttnn.reshape(x, (1, 1, math.prod(shape) // 1024, 1024), memory_config=ttnn.L1_MEMORY_CONFIG)
        cores = int(self.experiment.split("_")[1])
        if cores:
            grid = ttnn.CoreGrid(x=cores, y=1)
            mem = ttnn.create_sharded_memory_config(
                (grouped.padded_shape[-2], 1024 // cores),
                core_grid=grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            grouped = ttnn.to_memory_config(grouped, mem)
            cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(cores, 1),
                subblock_w=min(4, 32 // cores),
                block_h=grouped.padded_shape[-2] // 32,
                block_w=32 // cores,
                inplace=False,
            )
            grouped = ttnn.rms_norm(
                grouped, epsilon=self.eps, compute_kernel_config=self.compute, program_config=cfg, memory_config=mem
            )
        else:
            grouped = ttnn.rms_norm(
                grouped, epsilon=self.eps, compute_kernel_config=self.compute, memory_config=ttnn.L1_MEMORY_CONFIG
            )
        return ttnn.reshape(grouped, shape, memory_config=ttnn.L1_MEMORY_CONFIG)


class AttentionCandidate(OptimizedDecoder):
    experiment = "sdpa_8_8_128_l1"

    def decode_forward(self, *args, **kwargs):
        # A test-only, single-threaded adapter intercepts only the attention op.
        original = ttnn.transformer.paged_scaled_dot_product_attention_decode
        _, gx, gy, chunk, placement = self.experiment.split("_")

        def attention(*a, **kw):
            kw["program_config"] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(int(gx), int(gy)),
                q_chunk_size=0,
                k_chunk_size=int(chunk),
                exp_approx_mode=False,
            )
            kw["memory_config"] = ttnn.L1_MEMORY_CONFIG if placement == "l1" else ttnn.DRAM_MEMORY_CONFIG
            return original(*a, **kw)

        ttnn.transformer.paged_scaled_dot_product_attention_decode = attention
        try:
            return super().decode_forward(*args, **kwargs)
        finally:
            ttnn.transformer.paged_scaled_dot_product_attention_decode = original


def candidate_class(name):
    base = (
        PrefillCandidate
        if name.startswith("prefill_")
        else AttentionCandidate
        if name.startswith("sdpa_")
        else NormCandidate
    )
    return type(name, (base,), {"experiment": name})


class PrefillAttentionCandidate(OptimizedDecoder):
    experiment = "fastattention_128"

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        block = int(cls.experiment.split("_")[1])

        def attention(
            q, k, v, table, *, chunk_start_idx=None, chunk_start_idx_tensor=None, q_chunk_size=128, k_chunk_size=128
        ):
            return ttnn.transformer.chunked_scaled_dot_product_attention(
                q,
                k,
                v,
                table,
                chunk_start_idx=chunk_start_idx,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(11, 10),
                    q_chunk_size=min(block, q_chunk_size),
                    k_chunk_size=block,
                    exp_approx_mode=False,
                ),
                compute_kernel_config=obj.compute,
            )

        obj._attention = attention
        return obj


class PrefillNormCandidate(OptimizedDecoder):
    def _norm(self, x):
        if x.shape[2] <= 32:
            return super()._norm(x)
        shape = x.shape
        y = ttnn.reshape(x, (1, 1, math.prod(shape) // 1024, 1024))
        y = ttnn.rms_norm(
            y, epsilon=self.eps, compute_kernel_config=self.compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return ttnn.reshape(y, shape)


class PrefillL1Candidate(OptimizedDecoder):
    experiment = "l1prefill_all"

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        parts = cls.experiment.split("_")
        if len(parts) > 2 and parts[2].isdigit():
            obj.chunk_size = int(parts[2])
        return obj

    def _linear(self, x, w):
        group = self.experiment.split("_")[1]
        selected = (
            group == "all"
            or (group == "attention" and (w is self.wqkv or w is self.wo))
            or (group == "mlp" and (w is self.wgate or w is self.wup))
            or (group == "down" and w is self.wdown)
        )
        if x.shape[2] >= 256 and selected:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            if self.experiment.endswith("small"):
                return ttnn.experimental.minimal_matmul(
                    x,
                    w,
                    config=ttnn.MinimalMatmulConfig(
                        M_block_size=2,
                        K_block_size=4,
                        N_block_size=4,
                        subblock_h=2,
                        subblock_w=2,
                        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                    ),
                    compute_kernel_config=self._weight_compute(w),
                    dtype=ttnn.bfloat16,
                    **({"memory_config": ttnn.DRAM_MEMORY_CONFIG} if "dram" in self.experiment.split("_") else {}),
                )
        return super()._linear(x, w)


class SwigluCandidate(OptimizedDecoder):
    experiment = "swiglu_prefill"

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        obj = super().from_state_dict(state_dict, **kwargs)
        prefix = f"model.layers.{kwargs['layer_idx']}."
        gamma = state_dict[prefix + "post_attention_layernorm.weight"].bfloat16().float()
        weights = [
            (state_dict[prefix + "mlp." + p + "_proj.weight"].bfloat16().float().T * gamma[:, None]).bfloat16()
            for p in ["gate", "up"]
        ]
        # Each pair of adjacent tiles contains gate then up, the native SwiGLU contract.
        packed = torch.stack([w.reshape(4096, 384, 32) for w in weights], dim=2).reshape(4096, 24576)
        obj.swiglu = ttnn.from_torch(packed, device=obj.mesh_device, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
        return obj

    def _finish(self, x, attention):
        if x.shape[2] <= 32 and self.experiment == "swiglu_prefill":
            return super()._finish(x, attention)
        if x.shape[2] <= 32:
            proj = self._decode_linear(attention, self.wo)
            residual = ttnn.add(
                ttnn.to_memory_config(x, proj.memory_config()), proj, memory_config=proj.memory_config()
            )
        else:
            residual = ttnn.add(x, self._linear(attention, self.wo))
        normed = ttnn.to_memory_config(self._norm(residual), ttnn.DRAM_MEMORY_CONFIG)
        mlp = ttnn.experimental.minimal_matmul(
            normed,
            self.swiglu,
            config=ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=8,
                N_block_size=8,
                subblock_h=2,
                subblock_w=2,
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
            ),
            fuse_swiglu=True,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.role_compute["mlp"],
        )
        if x.shape[2] <= 32:
            down = self._decode_linear(mlp, self.wdown)
            return ttnn.add(
                ttnn.to_memory_config(residual, down.memory_config()), down, memory_config=down.memory_config()
            )
        return ttnn.add(residual, self._linear(mlp, self.wdown))


_old_candidate_class = candidate_class


def candidate_class(name):
    classes = {
        "fastattention": PrefillAttentionCandidate,
        "prefillnorm": PrefillNormCandidate,
        "l1prefill": PrefillL1Candidate,
        "swiglu": SwigluCandidate,
    }
    base = classes.get(name.split("_")[0])
    return type(name, (base,), {"experiment": name}) if base else _old_candidate_class(name)


class PrefillComputeCandidate(OptimizedDecoder):
    experiment = "prefillcompute_8_4"

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        for group in ["attention", "mlp", "down"]:
            obj.role_compute[group] = ttnn.init_device_compute_kernel_config(
                obj.mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        return obj

    def _linear(self, x, w):
        if x.shape[2] < 256:
            return super()._linear(x, w)
        _, block, sw = self.experiment.split("_")
        return ttnn.experimental.minimal_matmul(
            x,
            w,
            config=ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=int(block),
                N_block_size=8,
                subblock_h=2,
                subblock_w=int(sw),
                compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
            ),
            compute_kernel_config=self._weight_compute(w),
            dtype=ttnn.bfloat16,
        )


_previous_candidate_class = candidate_class


def candidate_class(name):
    if name.startswith("prefillcompute_"):
        return type(name, (PrefillComputeCandidate,), {"experiment": name})
    return _previous_candidate_class(name)


class SeparateQKVCandidate(OptimizedDecoder):
    """Tuned separate projections with the same packed head-split boundary."""

    experiment = "separateqkv_32_8_2"

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        obj = super().from_state_dict(state_dict, **kwargs)
        _, cores, block, readers = cls.experiment.split("_")
        cores, block, readers = int(cores), int(block), int(readers)
        prefix = f"model.layers.{kwargs['layer_idx']}."
        gamma = state_dict[prefix + "input_layernorm.weight"].float()
        mesh = obj.mesh_device
        banks = mesh.dram_grid_size()
        bg = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks.x - 1, banks.y - 1))})
        obj.parts_qkv = []
        for role in "qkv":
            tensor = (state_dict[prefix + f"self_attn.{role}_proj.weight"].float().T * gamma[:, None]).bfloat16()
            k, n = tensor.shape
            w = ttnn.from_torch(tensor.contiguous(), device=mesh, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
            obj.parts_qkv.append(w)
            # Q uses the selected O-like 4096-wide geometry; independently tune narrow K/V.
            c, b, r = (64, 8, 2) if role == "q" else (cores, block, readers)
            nphys = math.ceil(n / math.lcm(32 * c, 32 * banks.x * r)) * math.lcm(32 * c, 32 * banks.x * r)
            wm = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(bg, (k, nphys // banks.x), ttnn.ShardOrientation.ROW_MAJOR),
            )
            obj.decode_weights[id(w)] = ttnn.from_torch(
                torch.nn.functional.pad(tensor, (0, nphys - n)).contiguous(),
                device=mesh,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=wm,
            )
            grid = ttnn.num_cores_to_corerangeset(c, mesh.compute_with_storage_grid_size(), row_wise=True)
            obj.decode_inputs[id(w)] = ttnn.create_sharded_memory_config(
                (32, k // c),
                core_grid=grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            obj.decode_programs[id(w)] = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=b, per_core_M=1, per_core_N=nphys // (32 * c), num_workers_per_dram_bank=r
            )
            obj.decode_computes[id(w)] = obj.decode_computes[id(obj.wqkv)]
        grid = ttnn.num_cores_to_corerangeset(64, mesh.compute_with_storage_grid_size(), row_wise=True)
        obj.join_memory = ttnn.create_sharded_memory_config(
            (32, 96),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return obj

    def _linear(self, x, w):
        if w is not self.wqkv:
            return super()._linear(x, w)
        return ttnn.concat([super(SeparateQKVCandidate, self)._linear(x, p) for p in self.parts_qkv], dim=-1)

    def _decode_linear(self, x, w):
        if w is not self.wqkv:
            return super()._decode_linear(x, w)
        outputs = [
            ttnn.to_memory_config(super(SeparateQKVCandidate, self)._decode_linear(x, p), ttnn.L1_MEMORY_CONFIG)
            for p in self.parts_qkv
        ]
        return ttnn.to_memory_config(
            ttnn.concat(outputs, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG), self.join_memory
        )


_before_qkv_candidate_class = candidate_class


def candidate_class(name):
    if name.startswith("separateqkv_"):
        return type(name, (SeparateQKVCandidate,), {"experiment": name})
    return _before_qkv_candidate_class(name)


class FusedPrefillConfigCandidate(OptimizedDecoder):
    """Retune large blocks after selecting BF16 destination and native SwiGLU."""

    experiment = "fusedconfig_4_8_8"

    def _prefill_matmul_config(self):
        _, m, k, n = self.experiment.split("_")
        return ttnn.MinimalMatmulConfig(
            M_block_size=int(m),
            K_block_size=int(k),
            N_block_size=int(n),
            subblock_h=2,
            subblock_w=4,
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
        )


_before_fusedconfig_candidate_class = candidate_class


def candidate_class(name):
    if name.startswith("fusedconfig_"):
        return type(name, (FusedPrefillConfigCandidate,), {"experiment": name})
    return _before_fusedconfig_candidate_class(name)


class FusedL1PrefillCandidate(OptimizedDecoder):
    experiment = "fusedl1_4096"

    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        obj = super().from_state_dict(*args, **kwargs)
        obj.chunk_size = int(cls.experiment.split("_")[1])
        return obj

    def _finish(self, x, attention):
        if x.shape[2] < 256:
            return super()._finish(x, attention)
        original = ttnn.experimental.minimal_matmul

        def with_l1(a, b, **kw):
            if b is self.wswiglu:
                a = ttnn.to_memory_config(a, ttnn.L1_MEMORY_CONFIG)
                if self.experiment.endswith("dram"):
                    kw["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
                kw["config"] = ttnn.MinimalMatmulConfig(
                    M_block_size=4,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=2,
                    subblock_w=4,
                    compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                )
            return original(a, b, **kw)

        ttnn.experimental.minimal_matmul = with_l1
        try:
            return super()._finish(x, attention)
        finally:
            ttnn.experimental.minimal_matmul = original


_before_fusedl1_candidate_class = candidate_class


def candidate_class(name):
    if name.startswith("fusedl1_"):
        return type(name, (FusedL1PrefillCandidate,), {"experiment": name})
    return _before_fusedl1_candidate_class(name)
