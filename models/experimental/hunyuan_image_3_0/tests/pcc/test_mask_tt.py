# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TT custom attention mask verification (device).
#   - build_attention_mask_tt is built with TTNN ops only (no torch).
#   - Its keep/mask pattern must match the verified PyTorch reference
#     (which is itself bit-exact vs the upstream model), and masked entries
#     must be a large-negative additive value while kept entries are exactly 0.
#
# Run:
#   cd /home/iguser/ign-tt/tt-metal
#   python_env/bin/python models/experimental/hunyuan_image_3_0/tests/pcc/test_mask_tt.py

import sys
import torch  # test-only (reference + comparison); the TT module itself uses no torch

ROOT = "/home/iguser/ign-tt/tt-metal"
HUNYUAN = "/home/iguser/ign-tt/hunyan_instruct"
for p in (ROOT, HUNYUAN):
    if p not in sys.path:
        sys.path.insert(0, p)

import ttnn
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask
from models.experimental.hunyuan_image_3_0.tt.attention.mask import build_attention_mask_tt, _NEG

# This test validates the TT mask against the PyTorch reference (the single
# golden source). The reference is in turn proven bit-exact vs the actual
# upstream HunyuanImage-3.0 mask in test_mask_gen.py::test_ref_matches_upstream_bit_exact.


CASES = [
    ("causal only         S=32", 32, [[]]),
    ("one image span      S=32", 32, [[slice(4, 20)]]),
    ("two image spans     S=64", 64, [[slice(3, 11), slice(40, 56)]]),
    ("bsz=2 same spans    S=32", 32, [[slice(5, 12)], [slice(5, 12)]]),
    ("bsz=2 diff spans    S=32", 32, [[slice(2, 8)], [slice(16, 28)]]),
]


def ref_additive(seq_len, per_batch, bsz):
    """Golden additive mask, built from the ref bool mask using the SAME _NEG
    sentinel and bf16 dtype as the TT path, so a true element-wise bitwise
    comparison is meaningful (0.0 and _NEG are both exactly representable)."""
    ref_bool = build_attention_mask(seq_len, per_batch, bsz=bsz)  # [bsz,1,S,S] bool
    add = torch.where(
        ref_bool,
        torch.zeros((), dtype=torch.float32),
        torch.full((), _NEG, dtype=torch.float32),
    )
    return ref_bool, add.to(torch.bfloat16)  # match TT dtype for exact bits


def run(device):
    results = []
    for name, S, per_batch in CASES:
        bsz = len(per_batch)
        ref_bool, ref_add = ref_additive(S, per_batch, bsz)  # bf16 golden values

        tt = build_attention_mask_tt(device, S, per_batch, bsz=bsz)
        tt_t = ttnn.to_torch(tt)[..., :S, :S].to(torch.bfloat16)  # bf16 device values

        # FULL element-wise bitwise equality of the actual mask tensors,
        # not just shape/pattern. Both keep (0.0) and masked (_NEG) values
        # are exactly representable in bf16, so equality must be exact.
        bitwise_ok = torch.equal(tt_t, ref_add)
        shape_ok = tuple(tt_t.shape) == (bsz, 1, S, S)
        n_diff = int((tt_t != ref_add).sum().item())

        ok = bitwise_ok and shape_ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"         shape={tuple(tt_t.shape)} bitwise_equal={bitwise_ok} diff_elems={n_diff}")
        results.append(ok)
    return results


if __name__ == "__main__":
    print("Opening device …")
    device = ttnn.open_device(device_id=0)
    try:
        results = run(device)
    finally:
        ttnn.close_device(device)

    n = sum(results)
    print("\n" + "=" * 60)
    print(f"TT mask: {n}/{len(results)} PASSED")
    print("ALL PASS ✓" if n == len(results) else "SOME FAILED ✗")
    print("=" * 60)
    sys.exit(0 if n == len(results) else 1)
