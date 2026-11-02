# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 35 attribution of ttnn's matmul compute kernel (bmm_large_block_zm_fused_bias_activation_metal2.cpp, the Metal 2.0 fork the mcast factories bind; the legacy file beside it is not compiled): writes a
variant of the kernel from a pristine copy. The JIT build hashes the source, so the next run recompiles it.

    python iter35_attrib_patch.py <pristine> <target> <variant>   variant: base | nomath | nopack | nomath_nopack
"""

import sys

pristine, target, variant = sys.argv[1:4]
src = open(pristine).read()
if "nopack" in variant:
    for cb in ("mm_out_dfb_id", "mm_partials_dfb_id"):
        old = f"pack_block(start_dst_index, {cb}, out_subblock_num_tiles);"
        assert src.count(old) == 1, old
        src = src.replace(old, f"/* attrib: no pack */ (void)start_dst_index;")
if "nomath" in variant:
    src = src.replace("#include", "#define SKIP_COMPUTE 1\n#include", 1)
    assert src.count("#define SKIP_COMPUTE 1") == 1
open(target, "w").write(src)
print(variant, "written")
