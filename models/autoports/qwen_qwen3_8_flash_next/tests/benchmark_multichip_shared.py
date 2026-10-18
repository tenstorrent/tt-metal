"""Compare packed router/shared input projections and separate shared gate/up."""

import sys, math
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    ParallelMoE,
    _expert_kernel,
    _EXPERT_DISPATCH,
    _EXPERT_DISPATCH_SELECTED,
    _EXPERT_COMPACT,
    _EXPERT_DISPATCH_PACKED,
    _EXPERT_COMPACT_PACKED,
)


class PackedShared(ParallelMoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router = ttnn.concat([self.router, self.shared_gu], dim=-1)
        del self.shared_gu

    def group_native(self, x):
        n = x.shape[2]
        router = self.ops.linear(x, self.router, compute_kernel_config=self.shared_compute)
        logits = router[..., :512]
        values, indices = ttnn.topk(
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
        )
        selected = ttnn.softmax(
            ttnn.add(values, self.mask), dim=-1, numeric_stable=True, compute_kernel_config=self.ops.compute
        )
        ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        scores = ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT)
        counts = ttnn.empty((128,), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh)
        source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        _expert_kernel(self.ops.mesh, [source, ids, scores, self.rank, counts, self.capacity],
                       _EXPERT_DISPATCH_SELECTED, [n], workers=32, cb_size=5888)
        down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            self.capacity, self.offsets, counts, self.ids, self.gates, self.ups, self.downs, 32
        )
        compact = ttnn.empty(
            (1, 10, 32, 2560),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.ops.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _expert_kernel(
            self.ops.mesh, [down, ids, scores, self.rank, compact], _EXPERT_COMPACT, [n], workers=80, cb_size=8448
        )
        padded_scores = (
            ttnn.pad(selected[..., :10], [(0, 0), (0, 0), (0, 32 - n), (0, 0)], 0) if n < 32 else selected[..., :10]
        )
        weighted = ttnn.multiply(compact, ttnn.permute(padded_scores, (0, 3, 2, 1)), output_tensor=compact)
        result = ttnn.sum(weighted, dim=1, keepdim=True)[:, :, :n, :]
        shared = router[..., 544:]
        shared = self.ops.linear(
            ttnn.multiply(shared[..., :160], shared[..., 160:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.shared_compute,
        )
        return ttnn.add(
            result, ttnn.multiply(shared, router[..., 512:513], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        )

    def group_packed(self, x):
        n = x.shape[2]
        padded = math.ceil(n / 32) * 32
        router = self.ops.linear(x, self.router, compute_kernel_config=self.shared_compute)
        logits = router[..., :512]
        values, indices = ttnn.topk(
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
        )
        selected = ttnn.softmax(
            ttnn.add(values, self.mask), dim=-1, numeric_stable=True, compute_kernel_config=self.ops.compute
        )
        ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        scores = ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT)
        counts = ttnn.empty((128,), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh)
        mapping = ttnn.empty((1, 1, n, 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh)
        capacity = ttnn.empty(
            (1, 1, 128 * 128, 2560), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh
        )
        counts = ttnn.to_memory_config(counts, ttnn.DRAM_MEMORY_CONFIG)
        source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        _expert_kernel(
            self.ops.mesh,
            [source, ids, scores, self.rank, counts, mapping, capacity],
            _EXPERT_DISPATCH_PACKED,
            [n],
            workers=32,
            cb_size=5120 + n * 192 + 64,
        )
        down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            capacity, self.offsets128, counts, self.ids, self.gates, self.ups, self.downs, 128
        )
        compact = ttnn.empty(
            (1, 10, padded, 2560),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.ops.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _expert_kernel(
            self.ops.mesh,
            [down, ids, scores, self.rank, mapping, compact],
            _EXPERT_COMPACT_PACKED,
            [n],
            workers=80,
            cb_size=2176 + n * 320 + 64,
        )
        scores = (
            ttnn.pad(selected[..., :10], [(0, 0), (0, 0), (0, padded - n), (0, 0)], 0)
            if n < padded
            else selected[..., :10]
        )
        weighted = ttnn.multiply(compact, ttnn.permute(scores, (0, 3, 2, 1)), output_tensor=compact)
        result = ttnn.sum(weighted, dim=1, keepdim=True)[:, :, :n, :]
        shared = router[..., 544:]
        shared = self.ops.linear(
            ttnn.multiply(shared[..., :160], shared[..., 160:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.shared_compute,
        )
        return ttnn.add(
            result, ttnn.multiply(shared, router[..., 512:513], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        )


class SplitShared(ParallelMoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_gate = self.shared_gu[..., :160]
        self.shared_up = self.shared_gu[..., 160:]
        del self.shared_gu

    def group_native(self, x):
        n = x.shape[2]
        router = self.ops.linear(x, self.router, compute_kernel_config=self.shared_compute)
        logits = router[..., :512]
        values, indices = ttnn.topk(
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
        )
        selected = ttnn.softmax(
            ttnn.add(values, self.mask), dim=-1, numeric_stable=True, compute_kernel_config=self.ops.compute
        )
        ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        scores = ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT)
        counts = ttnn.empty((128,), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh)
        source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        _expert_kernel(self.ops.mesh, [source, ids, scores, self.rank, counts, self.capacity],
                       _EXPERT_DISPATCH_SELECTED, [n], workers=32, cb_size=5888)
        down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            self.capacity, self.offsets, counts, self.ids, self.gates, self.ups, self.downs, 32
        )
        compact = ttnn.empty(
            (1, 10, 32, 2560),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.ops.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _expert_kernel(
            self.ops.mesh, [down, ids, scores, self.rank, compact], _EXPERT_COMPACT, [n], workers=80, cb_size=8448
        )
        padded_scores = (
            ttnn.pad(selected[..., :10], [(0, 0), (0, 0), (0, 32 - n), (0, 0)], 0) if n < 32 else selected[..., :10]
        )
        weighted = ttnn.multiply(compact, ttnn.permute(padded_scores, (0, 3, 2, 1)), output_tensor=compact)
        result = ttnn.sum(weighted, dim=1, keepdim=True)[:, :, :n, :]
        gate = self.ops.linear(x, self.shared_gate, compute_kernel_config=self.shared_compute)
        up = self.ops.linear(x, self.shared_up, compute_kernel_config=self.shared_compute)
        shared = self.ops.linear(
            ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.shared_compute,
        )
        return ttnn.add(
            result, ttnn.multiply(shared, router[..., 512:513], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        )

    def group_packed(self, x):
        n = x.shape[2]
        padded = math.ceil(n / 32) * 32
        router = self.ops.linear(x, self.router, compute_kernel_config=self.shared_compute)
        logits = router[..., :512]
        values, indices = ttnn.topk(
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
        )
        selected = ttnn.softmax(
            ttnn.add(values, self.mask), dim=-1, numeric_stable=True, compute_kernel_config=self.ops.compute
        )
        ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        scores = ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT)
        counts = ttnn.empty((128,), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh)
        mapping = ttnn.empty((1, 1, n, 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh)
        capacity = ttnn.empty(
            (1, 1, 128 * 128, 2560), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh
        )
        counts = ttnn.to_memory_config(counts, ttnn.DRAM_MEMORY_CONFIG)
        source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        _expert_kernel(
            self.ops.mesh,
            [source, ids, scores, self.rank, counts, mapping, capacity],
            _EXPERT_DISPATCH_PACKED,
            [n],
            workers=32,
            cb_size=5120 + n * 192 + 64,
        )
        down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            capacity, self.offsets128, counts, self.ids, self.gates, self.ups, self.downs, 128
        )
        compact = ttnn.empty(
            (1, 10, padded, 2560),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.ops.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _expert_kernel(
            self.ops.mesh,
            [down, ids, scores, self.rank, mapping, compact],
            _EXPERT_COMPACT_PACKED,
            [n],
            workers=80,
            cb_size=2176 + n * 320 + 64,
        )
        scores = (
            ttnn.pad(selected[..., :10], [(0, 0), (0, 0), (0, padded - n), (0, 0)], 0)
            if n < padded
            else selected[..., :10]
        )
        weighted = ttnn.multiply(compact, ttnn.permute(scores, (0, 3, 2, 1)), output_tensor=compact)
        result = ttnn.sum(weighted, dim=1, keepdim=True)[:, :, :n, :]
        gate = self.ops.linear(x, self.shared_gate, compute_kernel_config=self.shared_compute)
        up = self.ops.linear(x, self.shared_up, compute_kernel_config=self.shared_compute)
        shared = self.ops.linear(
            ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.shared_compute,
        )
        return ttnn.add(
            result, ttnn.multiply(shared, router[..., 512:513], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        )


def main():
    from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_multichip_decoder as bench
    from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target

    i = sys.argv.index("--family")
    family = sys.argv[i + 1]
    del sys.argv[i : i + 2]
    target.ParallelMoE = {"packed": PackedShared, "split": SplitShared}[family]
    bench.CANDIDATE = dict(shared_family=family)
    bench.main()


if __name__ == "__main__":
    main()
