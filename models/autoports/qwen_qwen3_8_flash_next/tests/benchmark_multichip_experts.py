"""Whole-layer selected-expert placement/configuration experiments."""

import sys
import torch, ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_multichip_decoder as bench
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target


class TensorParallelMoE:
    """GPT-OSS-style selected-only sparse experts with TP4 intermediate slices."""

    def __init__(self, state, cfg, ops):
        import torch

        self.ops = ops
        gu = state["mlp.experts.gate_up_proj"]
        self.gu = ops.shards(
            [
                torch.cat((gu[:, r * 160 : (r + 1) * 160], gu[:, 640 + r * 160 : 640 + (r + 1) * 160]), 1)
                .transpose(-2, -1)
                .unsqueeze(0)
                for r in range(4)
            ],
            ttnn.bfloat4_b,
        )
        del gu
        down = state["mlp.experts.down_proj"]
        self.down = ops.shards(
            [down[:, :, r * 160 : (r + 1) * 160].transpose(-2, -1).unsqueeze(0) for r in range(4)], ttnn.bfloat4_b
        )
        self.router = ops.upload(
            torch.cat(
                (
                    state["mlp.gate.weight"].T,
                    torch.nn.functional.pad(state["mlp.shared_expert_gate.weight"].T, (0, 31)),
                ),
                -1,
            )
        )
        self.shared_gu = ops.shards(
            [
                torch.cat(
                    (
                        state["mlp.shared_expert.gate_proj.weight"][r * 160 : (r + 1) * 160].T,
                        state["mlp.shared_expert.up_proj.weight"][r * 160 : (r + 1) * 160].T,
                    ),
                    -1,
                )
                for r in range(4)
            ]
        )
        self.shared_down = ops.shards(
            [state["mlp.shared_expert.down_proj.weight"][:, r * 160 : (r + 1) * 160].T for r in range(4)]
        )
        mask = torch.full((1, 1, 1, 32), float("-inf"))
        mask[..., :10] = 0
        self.mask = ops.upload(mask)
        self.compute = ttnn.init_device_compute_kernel_config(
            ops.mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        self.shared_compute = ttnn.init_device_compute_kernel_config(
            ops.mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.programs = {}
        for width, cores, block in ((320, 10, 40), (2560, 40, 5)):
            self.programs[width] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(5, cores // 5),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=1,
                out_block_w=width // 32 // cores,
                per_core_M=1,
                per_core_N=width // 32 // cores,
                fuse_batch=False,
                mcast_in0=True,
            )

    def sparse(self, x, w, mask, width, input_sparse=False):
        return ttnn.sparse_matmul(
            x,
            w,
            sparsity=mask,
            nnz=None,
            is_input_a_sparse=input_sparse,
            is_input_b_sparse=True,
            program_config=self.programs[width],
            compute_kernel_config=self.compute,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def group(self, x):
        router = ttnn.linear(x, self.router, compute_kernel_config=self.shared_compute)
        logits = router[..., :512]
        values, indices = ttnn.topk(
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
        )
        selected = ttnn.softmax(
            ttnn.add(values, self.mask), dim=-1, numeric_stable=True, compute_kernel_config=self.ops.compute
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected)
        mask = ttnn.to_layout(ttnn.max(routing, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        gu = ttnn.reshape(self.sparse(x, self.gu, mask, 320), (1, 512, 32, 320))
        hidden = ttnn.multiply(gu[..., :160], gu[..., 160:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        down = self.sparse(hidden, self.down, mask, 2560, True)
        weighted = ttnn.multiply(down, ttnn.permute(routing, (0, 3, 2, 1)), output_tensor=down)
        result = ttnn.sum(weighted, dim=1, keepdim=True)
        shared = ttnn.linear(x, self.shared_gu, compute_kernel_config=self.shared_compute)
        shared = ttnn.linear(
            ttnn.multiply(shared[..., :160], shared[..., 160:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.shared_compute,
        )
        return ttnn.add(
            result, ttnn.multiply(shared, router[..., 512:513], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        )

    def __call__(self, x):
        b, s = tuple(x.shape)[1:3]
        lanes = []
        for lane in range(b):
            chunks = []
            for start in range(0, s, 32):
                valid = min(32, s - start)
                chunk = x[:, lane : lane + 1, start : start + valid, :]
                if valid < 32:
                    chunk = ttnn.pad(chunk, [(0, 0), (0, 0), (0, 32 - valid), (0, 0)], 0)
                chunks.append(self.group(chunk)[:, :, :valid, :])
            lanes.append(chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=2))
        result = lanes[0] if len(lanes) == 1 else ttnn.concat(lanes, dim=1)
        return self.ops.scatter_reduce(result)


class ExpertParallel(TensorParallelMoE):
    def __init__(self, state, cfg, ops):
        super().__init__(state, cfg, ops)
        # Candidate harness only: release TP tensors before constructing the
        # same-sized EP representation. Final implementation loads once.
        del self.gu, self.down
        gu = state["mlp.experts.gate_up_proj"]
        self.gu = ops.shards([v.transpose(-2, -1).unsqueeze(0) for v in gu.chunk(4, 0)], ttnn.bfloat4_b)
        del gu
        down = state["mlp.experts.down_proj"]
        self.down = ops.shards([v.transpose(-2, -1).unsqueeze(0) for v in down.chunk(4, 0)], ttnn.bfloat4_b)
        self.columns = ops.shards(
            [
                torch.arange(r * 128, (r + 1) * 128, dtype=torch.int32).reshape(1, 1, 1, 128).expand(1, 1, 32, 128)
                for r in range(4)
            ],
            ttnn.uint32,
        )
        for width, cores, block in ((1280, 20, 40), (2560, 40, 20)):
            n = width // 32 // cores
            self.programs[width] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(5, cores // 5),
                in0_block_w=block,
                out_subblock_h=1,
                out_subblock_w=min(n, 4),
                out_block_h=1,
                out_block_w=n,
                per_core_M=1,
                per_core_N=n,
                fuse_batch=False,
                mcast_in0=True,
            )

    def group(self, x):
        router = ttnn.linear(x, self.router, compute_kernel_config=self.shared_compute)
        logits = router[..., :512]
        values, indices = ttnn.topk(
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
        )
        selected = ttnn.softmax(
            ttnn.add(values, self.mask), dim=-1, numeric_stable=True, compute_kernel_config=self.ops.compute
        )
        routing = ttnn.gather(
            ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected), 3, index=self.columns
        )
        if getattr(self, "single_logical_row", False):
            routing = ttnn.pad(routing[:, :, :1, :], [(0, 0), (0, 0), (0, 31), (0, 0)], 0)
        mask = ttnn.to_layout(ttnn.max(routing, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        gu = ttnn.reshape(self.sparse(x, self.gu, mask, 1280), (1, 128, 32, 1280))
        hidden = ttnn.multiply(gu[..., :640], gu[..., 640:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        down = self.sparse(hidden, self.down, mask, 2560, True)
        weighted = ttnn.multiply(down, ttnn.permute(routing, (0, 3, 2, 1)), output_tensor=down)
        result = ttnn.sum(weighted, dim=1, keepdim=True)
        shared = ttnn.linear(x, self.shared_gu, compute_kernel_config=self.shared_compute)
        shared = ttnn.linear(
            ttnn.multiply(shared[..., :160], shared[..., 160:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.shared_compute,
        )
        return ttnn.add(
            result, ttnn.multiply(shared, router[..., 512:513], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        )


class PrunedExpertParallel(ExpertParallel):
    def __call__(self, x):
        self.single_logical_row = tuple(x.shape)[1:3] == (1, 1)
        return super().__call__(x)


class ExactTop10(TensorParallelMoE):
    """Binary route mask gives a proven exact count independent of score flush."""

    def __init__(self, state, cfg, ops):
        super().__init__(state, cfg, ops)
        flag = torch.zeros(1, 1, 1, 32)
        flag[..., :10] = 1
        self.active_ten = ops.upload(flag)

    def __call__(self, x):
        if tuple(x.shape)[1:3] != (1, 1):
            return super().__call__(x)
        router = ttnn.linear(x, self.router, compute_kernel_config=self.shared_compute)
        logits = router[..., :512]
        values, indices = ttnn.topk(
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
        )
        selected = ttnn.softmax(
            ttnn.add(values, self.mask), dim=-1, numeric_stable=True, compute_kernel_config=self.ops.compute
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected)
        binary = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=self.active_ten)
        mask = ttnn.to_layout(binary, ttnn.ROW_MAJOR_LAYOUT)
        padded = ttnn.pad(x, [(0, 0), (0, 0), (0, 31), (0, 0)], 0)

        def sparse(inp, w, width, input_sparse=False):
            return ttnn.sparse_matmul(
                inp,
                w,
                sparsity=mask,
                nnz=10,
                is_input_a_sparse=input_sparse,
                is_input_b_sparse=True,
                program_config=self.programs[width],
                compute_kernel_config=self.compute,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        gu = ttnn.reshape(sparse(padded, self.gu, 320), (1, 512, 32, 320))
        hidden = ttnn.multiply(gu[..., :160], gu[..., 160:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        down = sparse(hidden, self.down, 2560, True)
        scores = ttnn.pad(routing, [(0, 0), (0, 0), (0, 31), (0, 0)], 0)
        weighted = ttnn.multiply(down, ttnn.permute(scores, (0, 3, 2, 1)), output_tensor=down)
        result = ttnn.sum(weighted, dim=1, keepdim=True)[:, :, :1, :]
        shared = ttnn.linear(x, self.shared_gu, compute_kernel_config=self.shared_compute)
        shared = ttnn.linear(
            ttnn.multiply(shared[..., :160], shared[..., 160:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.shared_compute,
        )
        return self.ops.scatter_reduce(
            ttnn.add(
                result,
                ttnn.multiply(shared, router[..., 512:513], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]),
            )
        )


def main():
    bench.CANDIDATE = dict(command=list(sys.argv))
    index = sys.argv.index("--expert-placement")
    placement = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    bench.CANDIDATE["expert_placement"] = placement
    if placement == "ep":
        target.ParallelMoE = ExpertParallel
    elif placement == "ep_pruned":
        target.ParallelMoE = PrunedExpertParallel
    elif placement == "exact10":
        target.ParallelMoE = ExactTop10
    elif placement != "tp":
        raise ValueError(placement)
    bench.main()


if __name__ == "__main__":
    main()
