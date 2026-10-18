"""Whole-decoder topology controls; isolated processes pin each candidate.

These candidates cover GDN and QSA. PLE requires an adapted projection/conv
contract if either alternative wins; it cannot silently use hidden sharding.
"""

import sys
import torch, ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_multichip_decoder as bench
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedHyperConnection
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedDecoder


class ReplicatedHC(OptimizedHyperConnection):
    dram_readers = 0


class StreamHC:
    def __init__(self, state, prefix, cfg, ops):
        self.ops, self.eps = ops, cfg.rms_norm_eps
        self.norm = ops.shards(
            [v.reshape(1, 1, 1, 2560) for v in (1 + state[prefix + ".hc_norm.weight"].float()).chunk(4)]
        )
        packed = torch.cat(
            (
                state[prefix + ".input_mix_weight_down.weight"].T / 4,
                torch.nn.functional.pad(state[prefix + ".block_inject_weight.weight"].T / 4, (0, 28)),
            ),
            -1,
        )
        self.read_weight = ops.shards(list(packed.chunk(4, 0)), ttnn.bfloat8_b)
        self.up = ops.shards(list(state[prefix + ".input_mix_weight_up.weight"].T.chunk(4, 1)), ttnn.bfloat8_b)
        self.select = ops.shards(
            [torch.nn.functional.one_hot(torch.tensor(r), 4).reshape(1, 1, 1, 4).float() for r in range(4)]
        )
        self.compute = ttnn.init_device_compute_kernel_config(
            ops.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def read(self, x):
        normalized = ttnn.rms_norm(x, weight=self.norm, epsilon=self.eps, compute_kernel_config=self.ops.compute)
        packed = self.ops.reduce(ttnn.linear(normalized, self.read_weight, compute_kernel_config=self.compute))
        mix = ttnn.sigmoid(ttnn.linear(ttnn.silu(packed[..., :320]), self.up, compute_kernel_config=self.compute))
        mixed = self.ops.reduce(ttnn.multiply(ttnn.multiply(mix, normalized), 0.25))
        inject = ttnn.multiply(packed[..., 320:324], 2, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        return mixed, inject

    def write(self, x, update, inject):
        scale = ttnn.sum(ttnn.multiply(inject, self.select), dim=-1, keepdim=True)
        return ttnn.add(x, ttnn.multiply(update, scale))


def main():
    if "--native-experts" in sys.argv:
        sys.argv.remove("--native-experts")
        from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_native_experts import NativeExpertParallel

        target.ParallelMoE = NativeExpertParallel
    family = sys.argv[sys.argv.index("--residual-layout") + 1]
    bench.CANDIDATE = dict(command=list(sys.argv), residual_layout=family)
    layer = int(sys.argv[sys.argv.index("--layer") + 1])
    assert layer in [0, 3]
    target.MeshOps.scatter_reduce = target.MeshOps.reduce
    if family == "stream":
        target.ParallelHyperConnection = StreamHC
    elif family == "replicated":
        target.ParallelHyperConnection = lambda state, prefix, cfg, ops: ReplicatedHC.from_state_dict(
            state, prefix=prefix, hf_config=cfg, mesh_device=ops.mesh
        )
        target.MultichipDecoder.prefill_forward = FusedDecoder.prefill_forward
        target.MultichipDecoder.decode_forward = FusedDecoder.decode_forward
    else:
        raise ValueError("Use the unmodified benchmark for hidden sharding")
    bench.main()


if __name__ == "__main__":
    main()
