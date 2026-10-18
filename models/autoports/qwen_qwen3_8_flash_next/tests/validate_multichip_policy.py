"""Recorded boundary qualification for input-only BFP4 attention candidate."""

import sys
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target


class Input4GDN(target.ParallelGatedDeltaNet):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.packed_weight = ttnn.typecast(self.packed_weight, ttnn.bfloat4_b)
        self.input_compute = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _linear(self, x, weight):
        return self.ops.linear(
            x,
            weight,
            compute_kernel_config=self.input_compute if weight is self.packed_weight else self.projection_compute,
        )


class Input4QSA(target.ParallelQSA):
    @classmethod
    def build(cls, *a, **kw):
        obj = super().build(*a, **kw)
        obj.packed_projection = ttnn.typecast(obj.packed_projection, ttnn.bfloat4_b)
        obj.input_compute = ttnn.init_device_compute_kernel_config(
            obj.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        return obj

    def _linear(self, x, weight):
        return self.ops.linear(
            x,
            weight,
            compute_kernel_config=self.input_compute if weight is self.packed_projection else self.projection_compute,
        )


def main():
    target.ParallelGatedDeltaNet = Input4GDN
    target.ParallelQSA = Input4QSA
    from models.autoports.qwen_qwen3_8_flash_next.tests.validate_multichip_decoder import main

    main()


if __name__ == "__main__":
    main()
