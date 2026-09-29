"""Experimental graphs retained to reproduce measured fusion rejections."""

import math

import ttnn

from ..tt.functional_decoder import FunctionalDecoder
from .fusion_v1 import InitialFusion as FusedDecoder


class RopeOnly(FunctionalDecoder):
    _rope = staticmethod(FusedDecoder._rope)
    _decode_rope = FusedDecoder._decode_rope
    decode_forward = FusedDecoder.decode_forward
    _decode_qk = FusedDecoder._decode_qk
    _update_cache = FusedDecoder._update_cache


class WeightedNorm(RopeOnly):
    _norm = FusedDecoder._norm


class ReshapeNorm(FusedDecoder):
    def _norm(self, x, weight):
        shape = x.shape
        y = ttnn.reshape(x, (1, 1, math.prod(x.shape) // 1024, 1024))
        y = ttnn.rms_norm(y, epsilon=self.eps, compute_kernel_config=self.compute)
        return ttnn.multiply(ttnn.reshape(y, shape), weight)


class PackedGate(FusedDecoder):
    def _finish(self, x, attention):
        residual = ttnn.add(x, self._linear(attention, self.wo))
        normed = self._norm(residual, self.norm2)
        packed = self._linear(normed, self.wgateup)
        mlp = ttnn.multiply(
            packed[..., :12288], packed[..., 12288:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
        )
        return ttnn.add(residual, self._linear(mlp, self.wdown))


class LinearSilu(FusedDecoder):
    def _finish(self, x, attention):
        residual = ttnn.add(x, self._linear(attention, self.wo))
        normed = self._norm(residual, self.norm2)
        gate = ttnn.linear(
            normed,
            self.wgate,
            activation="silu",
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute,
        )
        mlp = ttnn.multiply(gate, self._linear(normed, self.wup))
        return ttnn.add(residual, self._linear(mlp, self.wdown))


CANDIDATES = dict(
    functional=FunctionalDecoder,
    rope=RopeOnly,
    norm=WeightedNorm,
    fused=FusedDecoder,
    reshape_norm=ReshapeNorm,
    packed=PackedGate,
    linear_silu=LinearSilu,
)


class FastRope(FusedDecoder):
    _norm = FunctionalDecoder._norm

    def _decode_qk(self, q, k, rope):
        if q.shape[1] == 1:
            q = ttnn.experimental.rotary_embedding(q, rope[0], rope[1], token_index=0, memory_config=q.memory_config())
            k = ttnn.experimental.rotary_embedding(k, rope[0], rope[1], token_index=0, memory_config=k.memory_config())
            return ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG), k
        return super()._decode_qk(q, k, rope)


class FusedCache(FastRope):
    def _update_cache(self, kv_cache, k, v, current_pos, page_table):
        batch = k.shape[1]
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
        available = grid.subtract(k.memory_config().shard_spec.grid)
        vg = ttnn.num_cores_to_corerangeset_in_subcoregrids(available.ranges()[0].start, batch, available, True)
        vs = ttnn.create_sharded_memory_config(
            (32, 128),
            core_grid=vg,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        v = ttnn.to_memory_config(v, vs)
        ttnn.experimental.paged_fused_update_cache(
            kv_cache[0], k, kv_cache[1], v, update_idxs_tensor=current_pos, page_table=page_table
        )


class FastReshape(FastRope):
    _norm = ReshapeNorm._norm


class ResidualNorm(FastRope):
    def _finish(self, x, attention):
        projected = self._linear(attention, self.wo)
        residual = ttnn.add(x, projected)
        groups = [
            ttnn.rms_norm(
                x[..., i : i + 1024],
                residual_input_tensor=projected[..., i : i + 1024],
                epsilon=self.eps,
                weight=w,
                compute_kernel_config=self.compute,
            )
            for i, w in zip(range(0, 4096, 1024), self.norm_groups[id(self.norm2)])
        ]
        normed = ttnn.concat(groups, dim=-1)
        mlp = ttnn.multiply(
            self._linear(normed, self.wgate),
            self._linear(normed, self.wup),
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )
        return ttnn.add(residual, self._linear(mlp, self.wdown))


CANDIDATES.update(fast_rope=FastRope, fused_cache=FusedCache, fast_reshape=FastReshape, residual_norm=ResidualNorm)


class Combined(FusedCache):
    def _norm(self, x, weight):
        if x.shape[2] <= 32:
            return ReshapeNorm._norm(self, x, weight)
        return FunctionalDecoder._norm(self, x, weight)


class CombinedPacked(Combined):
    _finish = PackedGate._finish


class CombinedLinear(Combined):
    _finish = LinearSilu._finish


class ExpertMLP(Combined):
    def _finish(self, x, attention):
        residual = ttnn.add(x, self._linear(attention, self.wo))
        normed = self._norm(residual, self.norm2)
        rows = normed.shape[2]
        padded = ttnn.pad(normed, [(0, 0), (0, 0), (0, (-rows) % 32), (0, 0)], 0) if rows % 32 else normed
        rm = ttnn.to_layout(padded, ttnn.ROW_MAJOR_LAYOUT)
        y = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            rm,
            [self.wgate],
            [self.wup],
            [self.wdown],
            self.expert_counts[rows],
            self.expert_ids,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.expert_compute,
            core_grid=ttnn.CoreCoord(8, 8),
        )
        y = ttnn.to_layout(y, ttnn.TILE_LAYOUT)[:, :, :rows, :]
        return ttnn.add(residual, y)


CANDIDATES.update(
    combined=Combined, combined_packed=CombinedPacked, combined_linear=CombinedLinear, expert_mlp=ExpertMLP
)


class StockDecode(CombinedPacked):
    def decode_forward(self, x, *, rope, kv_cache, page_table, current_pos):
        batch = x.shape[2]
        qkv = self._linear(self._norm(x, self.norm1), self.wqkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=32, num_kv_heads=8, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        q, k = self._decode_qk(q, k, rope)
        grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
        shard = ttnn.create_sharded_memory_config(
            (32, 128),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        k = ttnn.to_memory_config(k, shard)
        v = ttnn.to_memory_config(v, shard)
        self._update_cache(kv_cache, k, v, current_pos, page_table)
        attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            compute_kernel_config=self.compute,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=128, exp_approx_mode=False
            ),
        )
        attention = ttnn.experimental.nlp_concat_heads_decode(ttnn.to_memory_config(attention, shard), num_heads=32)
        attention = ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG)[:, :, :batch, :]
        return self._finish(x, attention)


CANDIDATES.update(stock_decode=StockDecode)


class StockSeparate(StockDecode):
    _finish = FusedDecoder._finish


class ExpertDecode(StockDecode):
    def _finish(self, x, attention):
        if x.shape[2] > 32:
            return Combined._finish(self, x, attention)
        return ExpertMLP._finish(self, x, attention)


CANDIDATES.update(stock_separate=StockSeparate, expert_decode=ExpertDecode)


class DirectConcat(StockDecode):
    def decode_forward(self, x, *, rope, kv_cache, page_table, current_pos):
        batch = x.shape[2]
        qkv = self._linear(self._norm(x, self.norm1), self.wqkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=32, num_kv_heads=8, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        q, k = self._decode_qk(q, k, rope)
        grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
        shard = ttnn.create_sharded_memory_config(
            (32, 128),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        k = ttnn.to_memory_config(k, shard)
        v = ttnn.to_memory_config(v, shard)
        self._update_cache(kv_cache, k, v, current_pos, page_table)
        attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            compute_kernel_config=self.compute,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=128, exp_approx_mode=False
            ),
        )
        attention = ttnn.reshape(attention, (1, 1, batch, 4096))
        return self._finish(x, attention)


class DirectQ(DirectConcat):
    def _decode_qk(self, q, k, rope):
        if q.shape[1] == 1:
            return tuple(
                ttnn.experimental.rotary_embedding(t, rope[0], rope[1], token_index=0, memory_config=t.memory_config())
                for t in (q, k)
            )
        return super()._decode_qk(q, k, rope)


CANDIDATES.update(direct_concat=DirectConcat, direct_q=DirectQ)


# Final default is measured alongside the archived best candidates.
from ..tt.fused_decoder import FusedDecoder as DeliveredDecoder

CANDIDATES["delivered"] = DeliveredDecoder


class ShardedProjection(DirectQ):
    def decode_forward(self, x, *, rope, kv_cache, page_table, current_pos):
        batch = x.shape[2]
        cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 6),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=4,
            fuse_batch=True,
            mcast_in0=True,
        )
        qkv = ttnn.linear(
            self._norm(x, self.norm1),
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.compute,
            program_config=cfg,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=32,
            num_kv_heads=8,
            # Split-half rotary's factory requires origin-prefix shard grids.
            overlap_qk_coregrid=True,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        q, k = self._decode_qk(q, k, rope)
        grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
        shard = ttnn.create_sharded_memory_config(
            (32, 128),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self._update_cache(kv_cache, k, v, current_pos, page_table)
        attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            compute_kernel_config=self.compute,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=128, exp_approx_mode=False
            ),
        )
        attention = ttnn.reshape(attention, (1, 1, batch, 4096))
        return self._finish(x, attention)


CANDIDATES["sharded_projection"] = ShardedProjection


class ShardedProjection96(ShardedProjection):
    def decode_forward(self, x, *, rope, kv_cache, page_table, current_pos):
        batch = x.shape[2]
        cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(11, 9),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=2,
            fuse_batch=True,
            mcast_in0=True,
        )
        qkv = ttnn.linear(
            self._norm(x, self.norm1),
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.compute,
            program_config=cfg,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=32,
            num_kv_heads=8,
            overlap_qk_coregrid=True,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        q, k = self._decode_qk(q, k, rope)
        grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
        shard = ttnn.create_sharded_memory_config(
            (32, 128),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self._update_cache(kv_cache, k, v, current_pos, page_table)
        attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_cache[0],
            kv_cache[1],
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            compute_kernel_config=self.compute,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=128, exp_approx_mode=False
            ),
        )
        attention = ttnn.reshape(attention, (1, 1, batch, 4096))
        return self._finish(x, attention)


CANDIDATES["sharded_projection96"] = ShardedProjection96


class DynamicResidual(DirectQ):
    def _finish(self, x, attention):
        if x.shape[2] != 1:
            return CombinedPacked._finish(self, x, attention)
        residual = ttnn.linear(
            attention,
            self.wo,
            bias=x,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute,
        )
        normed = self._norm(residual, self.norm2)
        packed = self._linear(normed, self.wgateup)
        mlp = ttnn.multiply(
            packed[..., :12288], packed[..., 12288:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
        )
        return ttnn.linear(
            mlp,
            self.wdown,
            bias=residual,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute,
        )


class ShardedResidual(ShardedProjection):
    _finish = DynamicResidual._finish


class ShardedResidual96(ShardedProjection96):
    _finish = DynamicResidual._finish


CANDIDATES.update(
    dynamic_residual=DynamicResidual, sharded_residual=ShardedResidual, sharded_residual96=ShardedResidual96
)


from .fusion_v2 import StageFusion


class VectorRope(StageFusion):
    def _decode_qk(self, q, k, rope):
        batch = q.shape[1]
        if batch == 1:
            return super()._decode_qk(q, k, rope)
        # Reinterpret batch as the rotary sequence dimension, keeping each
        # request's angle distinct while broadcasting it over all heads.
        rr = tuple(ttnn.reshape(r[:, :, :1, :], (1, 1, batch, 128)) for r in rope)
        outputs = []
        for t in (q, k):
            t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
            t = ttnn.permute(t, (0, 2, 1, 3))
            t = ttnn.experimental.rotary_embedding(t, rr[0], rr[1])[:, :, :batch, :]
            outputs.append(ttnn.permute(t, (0, 2, 1, 3)))
        return tuple(outputs)


CANDIDATES["vector_rope"] = VectorRope


class FoldNormAffine(StageFusion):
    def configure(self):
        for name in ("wqkv", "wgate", "wup", "wgateup"):
            setattr(self, name, getattr(self, "folded_" + name))

    def _norm(self, x, weight):
        if x.shape[2] <= 32:
            shape = x.shape
            grouped = ttnn.reshape(x, (1, 1, math.prod(shape) // 1024, 1024))
            grouped = ttnn.rms_norm(grouped, epsilon=self.eps, compute_kernel_config=self.compute)
            return ttnn.reshape(grouped, shape)
        return ttnn.concat(
            [
                ttnn.rms_norm(x[..., i : i + 1024], epsilon=self.eps, compute_kernel_config=self.compute)
                for i in range(0, 4096, 1024)
            ],
            dim=-1,
        )


CANDIDATES["fold_norm_affine"] = FoldNormAffine


class FoldNormResidual(FoldNormAffine):
    _finish = DynamicResidual._finish


CANDIDATES["fold_norm_residual"] = FoldNormResidual


class FoldNormShardedResidual96(FoldNormResidual):
    decode_forward = ShardedProjection96.decode_forward


CANDIDATES["fold_norm_sharded_residual96"] = FoldNormShardedResidual96

from .fusion_before_v import BeforeVDecoder

CANDIDATES["before_v"] = BeforeVDecoder
