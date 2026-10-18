# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY: six separate-rounded product/fold traversals per resident token."""

import hashlib
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_resident_candidate as qualified

PINS = {
    **qualified.PINS,
    "benchmark_gdn_resident_candidate.py": "f6be388cf6da89452127b8b4bf11918783b0e1d07dd89fa0c362045032f65cfc",
    "benchmark_gdn_resident_compute.cpp": "dddc593e27aac0d554ae4f5cb5cc37a8bf0d110b42f65e3e8e555c4c8dd14bb2",
    "benchmark_gdn_resident_probe.py": "deda247ad3f41389b8073f466412768c16ee07f962b399a86ec06bd63da95870",
}
for name, expected in PINS.items():
    if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Qualified resident source changed: {name}")

REPO = qualified.REPO
NATIVE_PINS = {
    **qualified.NATIVE_PINS,
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h": "3d78bbd9b0c962fca44423e69be1adc46727bed22a319c4c4acd2a7c65b20eee",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_quant.h": "2e967360a9bfe9c2900222f5f658661d6ec033fa1e9df5760dff3e464eeff8c8",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu_params.h": "d8bd39a80167608a2241b4802fea2bf4cd509cebde26bf719dc39f2ce67c784e",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu.h": "5c532596b6e52d127f644bf17c2a7e2ffa25e7913f7d58c4ca00fc1c961e5e44",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h": "f4cc7bef24b36bdbd341c664adfc4261d62ec8b8156ebabc8a218ca88685e4cf",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h": "39617c00bea8653c78763aea601c88de8f5203b5e48fe97a6a96b8325bd72b91",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/cmath_common.h": "eecf10004fd44910f967a6481f2727f0758b24a6841c0dee33d443c1ece13098",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary_bcast.h": "28bf9237ad07c5e71ad587a488c0c5927085474d9186c08f441062bfc0f85584",
    "runtime/sfpi/include/sfpi_classes.h": "76ef7a7b563d8fab040c8f9e16e676b1eea823455d75c92b7819452b5b7e57ec",
    "runtime/sfpi/include/sfpi_funcs.h": "8c0b67f89a9cc10971dad253b3746108dd7e8ad5c4955806f2322cbbb06543bc",
}
for name, expected in NATIVE_PINS.items():
    if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Native separate-rounding/address contract changed: {name}")
baseline = qualified.baseline
HEADER_PATH = Path(__file__).with_name("benchmark_gdn_fused_fold_compute.hpp")
BODY_PATH = qualified.BODY_PATH
READER = qualified.READER


def transform_body(source):
    """Require exactly the two original four-K-tile loops; alter only i>0 folds."""
    old = """                half_mul(i, 6, i == 0 ? 4 : 5);
                if (i != 0) {
                    add_binary_tile_init();
                    half_add(4, 5, 4);
                }"""
    new = """                if (i == 0) {
                    half_mul(i, 6, 4);
                } else {
                    half_product_fold(i, 6, 4);
                }"""
    if source.count(old) != 2:
        raise ValueError("Expected exactly two qualified K-tile product/fold loops")
    result = source.replace(old, new)
    if result.count("half_product_fold(i, 6, 4);") != 2:
        raise ValueError("Unexpected product/fold replacement count")
    return result


body = BODY_PATH.read_text()
if qualified.COMPUTE.count(body) != 1:
    raise ValueError("Qualified resident body must occur exactly once")
COMPUTE = qualified.COMPUTE.replace(body, HEADER_PATH.read_text() + "\n" + transform_body(body))
LAUNCHER_SOURCE = qualified.LAUNCHER_SOURCE
_namespace = dict(vars(baseline), READER=READER, COMPUTE=COMPUTE)
exec(compile(LAUNCHER_SOURCE, __file__, "exec"), _namespace)
RECURRENCE = _namespace["resident_recurrence"]
RECURRENCES = {
    "current_rm": baseline._rm_recurrence,
    "resident_zero": qualified.RECURRENCE,
    "fold_zero": RECURRENCE,
    "fold_nan": RECURRENCE,
}
gdn_method = qualified.gdn_method
