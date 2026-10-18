"""Isolate GDN projection fidelity without changing state or collective code."""
import argparse
import sys
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as target


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--gdn-role', choices=['all', 'input', 'output', 'prefill_input', 'prefill_all'], default='all')
    args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    original = target.ParallelGatedDeltaNet
    class GDN(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.prefill_mode = True
            self.hifi2 = ttnn.init_device_compute_kernel_config(self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        def _linear(self, x, weight):
            use = args.gdn_role == 'all' or (args.gdn_role == 'input' and weight is self.packed_weight) or (args.gdn_role == 'output' and weight is self.out_weight) or (args.gdn_role == 'prefill_input' and self.prefill_mode and weight is self.packed_weight) or (args.gdn_role == 'prefill_all' and self.prefill_mode)
            return self.ops.linear(x, weight, compute_kernel_config=self.hifi2 if use else self.projection_compute)
        def prefill_forward(self, *a, **kw):
            self.prefill_mode = True
            return super().prefill_forward(*a, **kw)
        def decode_forward(self, *a, **kw):
            self.prefill_mode = False
            return super().prefill_forward(*a, **kw)
    target.ParallelGatedDeltaNet = GDN
    from models.autoports.qwen_qwen3_8_flash_next.tests.validate_multichip_decoder import main
    main()

if __name__ == '__main__':
    main()
