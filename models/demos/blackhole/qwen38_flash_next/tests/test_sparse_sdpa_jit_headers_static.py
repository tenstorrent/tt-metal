# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
QSA = REPO_ROOT / "models/demos/blackhole/qwen38_flash_next/ttnn/qsa.py"
COMPUTE_STREAMING = REPO_ROOT / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"
INDEXER_SCORE_COMPUTE = (
    REPO_ROOT / "ttnn/cpp/ttnn/operations/experimental/indexer_score/device/kernels/compute_indexer_score.cpp"
)
TOPK_LARGE_INDICES_COMPUTE = (
    REPO_ROOT / "ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/compute.cpp"
)
JIT_HEADER_MANIFEST = REPO_ROOT / "tt_metal/hw/sources.cmake"
JIT_API_ROOT = REPO_ROOT / "tt_metal/hw/inc"
EXPERIMENTAL_CCL_CMAKE = REPO_ROOT / "ttnn/cpp/ttnn/operations/experimental/ccl/CMakeLists.txt"
MOE_COMPUTE_KERNEL = REPO_ROOT / ("ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/compute.cpp")
MOE_RING_HEADER = REPO_ROOT / ("ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/moe_ring_common.h")
MOE_CONFIG_HEADER = REPO_ROOT / (
    "ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/hostdevcommon/config.hpp"
)
MOE_SWIGLU_HEADER = REPO_ROOT / ("ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/swiglu_sfpu.h")
REQUIRED_STREAMING_HEADERS = (
    "api/compute/experimental/matmul_custom.h",
    "api/compute/experimental/sdpa_sub_custom.h",
)
STREAMING_SDPA_BLACKHOLE_EXTENSION_CLOSURE = (
    "tt_metal/hw/inc/api/compute/experimental/matmul_custom.h",
    "tt_metal/hw/inc/api/compute/experimental/sdpa_sub_custom.h",
    "tt_metal/hw/inc/api/compute/common.h",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_math_matmul_custom_api.h",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_math_eltwise_binary_custom_api.h",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_unpack_AB_sub_bcast_col_custom_api.h",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_matmul_custom_no_mop.h",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_eltwise_binary_custom.h",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_unpack_AB_sub_bcast_col_custom.h",
)
INDEXER_SCORE_API_HEADER = "api/compute/experimental/indexer_mul_custom.h"
INDEXER_SCORE_RUNTIME_CLOSURE = (
    "tt_metal/hw/inc/api/compute/experimental/indexer_mul_custom.h",
    "tt_metal/hw/inc/api/compute/common.h",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_math_eltwise_binary_custom_api.h",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_unpack_AB_sub_bcast_col_custom_api.h",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_eltwise_binary_custom.h",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_unpack_AB_sub_bcast_col_custom.h",
)
TOPK_LARGE_INDICES_API_HEADER = "api/compute/experimental/topk_xl.h"
TOPK_LARGE_INDICES_BLACKHOLE_EXTENSION_CLOSURE = (
    "tt_metal/hw/inc/api/compute/experimental/topk_xl.h",
    "tt_metal/hw/inc/api/compute/compute_kernel_api.h",
    "tt_metal/hw/inc/api/compute/common.h",
    "tt_metal/hw/inc/api/compute/tile_move_copy.h",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_unpack_A_topk_xl_copy_api.h",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_math_topk_xl_copy_api.h",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/llk_math_eltwise_unary_sfpu_topk_xl.h",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_unpack_A_topk_xl_copy.h",
    "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/experimental/llk_math_eltwise_unary_datacopy_topk_xl_copy.h",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_topk_xl.h",
    "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_set_dst_write_addr_offset.h",
)
TOKEN3_QSA_SELECTION_BLACKHOLE_EXTENSION_CLOSURE = tuple(
    dict.fromkeys(
        STREAMING_SDPA_BLACKHOLE_EXTENSION_CLOSURE
        + INDEXER_SCORE_RUNTIME_CLOSURE
        + TOPK_LARGE_INDICES_BLACKHOLE_EXTENSION_CLOSURE
    )
)


def _assert_indexer_score_runtime_closure(runtime_root: Path) -> None:
    missing = [relative for relative in INDEXER_SCORE_RUNTIME_CLOSURE if not (runtime_root / relative).is_file()]
    assert not missing, f"missing indexer_score runtime header closure: {missing}"
    for relative in INDEXER_SCORE_RUNTIME_CLOSURE:
        assert (runtime_root / relative).read_bytes() == (REPO_ROOT / relative).read_bytes()


def _assert_topk_large_indices_blackhole_extension_closure(runtime_root: Path) -> None:
    missing = [
        relative
        for relative in TOPK_LARGE_INDICES_BLACKHOLE_EXTENSION_CLOSURE
        if not (runtime_root / relative).is_file()
    ]
    assert not missing, f"missing topk_large_indices Blackhole runtime header closure: {missing}"
    for relative in TOPK_LARGE_INDICES_BLACKHOLE_EXTENSION_CLOSURE:
        assert (runtime_root / relative).read_bytes() == (REPO_ROOT / relative).read_bytes(), relative


def _assert_token3_qsa_selection_blackhole_extension_closure(runtime_root: Path) -> None:
    missing = [
        relative
        for relative in TOKEN3_QSA_SELECTION_BLACKHOLE_EXTENSION_CLOSURE
        if not (runtime_root / relative).is_file()
    ]
    assert not missing, f"missing token3 QSA selection Blackhole runtime header closure: {missing}"
    for relative in TOKEN3_QSA_SELECTION_BLACKHOLE_EXTENSION_CLOSURE:
        assert (runtime_root / relative).read_bytes() == (REPO_ROOT / relative).read_bytes(), relative


def test_streaming_sdpa_experimental_headers_are_in_jit_install_manifest() -> None:
    compute_streaming = COMPUTE_STREAMING.read_text(encoding="utf-8")
    manifest = JIT_HEADER_MANIFEST.read_text(encoding="utf-8")

    for header in REQUIRED_STREAMING_HEADERS:
        assert f'#include "{header}"' in compute_streaming
        assert (JIT_API_ROOT / header).is_file()
        assert manifest.count(f"inc/{header}") == 1


def test_indexer_score_experimental_header_is_in_jit_install_manifest_with_blackhole_closure() -> None:
    compute = INDEXER_SCORE_COMPUTE.read_text(encoding="utf-8")
    api_header = (JIT_API_ROOT / INDEXER_SCORE_API_HEADER).read_text(encoding="utf-8")
    manifest = JIT_HEADER_MANIFEST.read_text(encoding="utf-8")

    assert f'#include "{INDEXER_SCORE_API_HEADER}"' in compute
    assert manifest.count(f"inc/{INDEXER_SCORE_API_HEADER}") == 1
    assert '#include "api/compute/common.h"' in api_header
    assert "#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)" in api_header
    assert "#if defined(TRISC_UNPACK) && defined(ARCH_BLACKHOLE)" in api_header
    assert '#include "experimental/llk_math_eltwise_binary_custom_api.h"' in api_header
    assert '#include "experimental/llk_unpack_AB_sub_bcast_col_custom_api.h"' in api_header
    assert '#include "experimental/llk_math_eltwise_binary_custom.h"' in (
        REPO_ROOT / INDEXER_SCORE_RUNTIME_CLOSURE[2]
    ).read_text(encoding="utf-8")
    assert '#include "experimental/llk_unpack_AB_sub_bcast_col_custom.h"' in (
        REPO_ROOT / INDEXER_SCORE_RUNTIME_CLOSURE[3]
    ).read_text(encoding="utf-8")
    _assert_indexer_score_runtime_closure(REPO_ROOT)


def test_indexer_score_runtime_closure_gate_rejects_omitted_api_header(tmp_path: Path, expect_error) -> None:
    for relative in INDEXER_SCORE_RUNTIME_CLOSURE[1:]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)

    with expect_error(AssertionError, "indexer_mul_custom.h"):
        _assert_indexer_score_runtime_closure(tmp_path)

    api_destination = tmp_path / INDEXER_SCORE_RUNTIME_CLOSURE[0]
    api_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / INDEXER_SCORE_RUNTIME_CLOSURE[0], api_destination)
    _assert_indexer_score_runtime_closure(tmp_path)


def test_token3_qsa_selection_custom_kernels_are_bounded() -> None:
    qsa = QSA.read_text(encoding="utf-8")
    compute_streaming = COMPUTE_STREAMING.read_text(encoding="utf-8")
    indexer_compute = INDEXER_SCORE_COMPUTE.read_text(encoding="utf-8")
    topk_compute = TOPK_LARGE_INDICES_COMPUTE.read_text(encoding="utf-8")

    assert "ttnn.experimental.indexer_score_dsa(" in qsa
    assert "ttnn.experimental.topk_large_indices(" in qsa
    assert "ttnn.transformer.sparse_sdpa(" in qsa
    assert f'#include "{INDEXER_SCORE_API_HEADER}"' in indexer_compute
    assert f'#include "{TOPK_LARGE_INDICES_API_HEADER}"' in topk_compute
    for header in REQUIRED_STREAMING_HEADERS:
        assert f'#include "{header}"' in compute_streaming


def test_topk_large_indices_header_is_in_jit_manifest_with_complete_blackhole_extension_closure() -> None:
    compute = TOPK_LARGE_INDICES_COMPUTE.read_text(encoding="utf-8")
    api_header = (JIT_API_ROOT / TOPK_LARGE_INDICES_API_HEADER).read_text(encoding="utf-8")
    manifest = JIT_HEADER_MANIFEST.read_text(encoding="utf-8")

    assert f'#include "{TOPK_LARGE_INDICES_API_HEADER}"' in compute
    assert manifest.count(f"inc/{TOPK_LARGE_INDICES_API_HEADER}") == 1
    assert '#include "api/compute/compute_kernel_api.h"' in api_header
    assert '#include "api/compute/common.h"' in api_header
    assert '#include "api/compute/tile_move_copy.h"' in api_header
    assert api_header.count("#ifdef ARCH_BLACKHOLE") == 3
    assert '#include "experimental/llk_unpack_A_topk_xl_copy_api.h"' in api_header
    assert '#include "experimental/llk_math_topk_xl_copy_api.h"' in api_header
    assert '#include "experimental/llk_sfpu/llk_math_eltwise_unary_sfpu_topk_xl.h"' in api_header
    assert '#include "experimental/llk_unpack_A_topk_xl_copy.h"' in (
        REPO_ROOT / TOPK_LARGE_INDICES_BLACKHOLE_EXTENSION_CLOSURE[4]
    ).read_text(encoding="utf-8")
    assert '#include "experimental/llk_math_eltwise_unary_datacopy_topk_xl_copy.h"' in (
        REPO_ROOT / TOPK_LARGE_INDICES_BLACKHOLE_EXTENSION_CLOSURE[5]
    ).read_text(encoding="utf-8")
    assert '#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"' in (
        REPO_ROOT / TOPK_LARGE_INDICES_BLACKHOLE_EXTENSION_CLOSURE[6]
    ).read_text(encoding="utf-8")
    assert '#include "sfpu/experimental/ckernel_sfpu_set_dst_write_addr_offset.h"' in (
        REPO_ROOT / TOPK_LARGE_INDICES_BLACKHOLE_EXTENSION_CLOSURE[9]
    ).read_text(encoding="utf-8")
    _assert_topk_large_indices_blackhole_extension_closure(REPO_ROOT)


def test_topk_large_indices_closure_gate_rejects_every_omission_and_byte_mismatch(tmp_path: Path, expect_error) -> None:
    for relative in TOPK_LARGE_INDICES_BLACKHOLE_EXTENSION_CLOSURE:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)

    _assert_topk_large_indices_blackhole_extension_closure(tmp_path)
    for relative in TOPK_LARGE_INDICES_BLACKHOLE_EXTENSION_CLOSURE:
        destination = tmp_path / relative
        original = destination.read_bytes()
        destination.unlink()
        with expect_error(AssertionError, "topk_large_indices"):
            _assert_topk_large_indices_blackhole_extension_closure(tmp_path)
        destination.write_bytes(original)

    mismatch = tmp_path / TOPK_LARGE_INDICES_BLACKHOLE_EXTENSION_CLOSURE[-1]
    mismatch.write_bytes(mismatch.read_bytes() + b"\n")
    with expect_error(AssertionError, TOPK_LARGE_INDICES_BLACKHOLE_EXTENSION_CLOSURE[-1]):
        _assert_topk_large_indices_blackhole_extension_closure(tmp_path)


def test_token3_qsa_selection_closure_gate_rejects_every_omission_and_byte_mismatch(
    tmp_path: Path, expect_error
) -> None:
    for relative in TOKEN3_QSA_SELECTION_BLACKHOLE_EXTENSION_CLOSURE:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)

    _assert_token3_qsa_selection_blackhole_extension_closure(tmp_path)
    for relative in TOKEN3_QSA_SELECTION_BLACKHOLE_EXTENSION_CLOSURE:
        destination = tmp_path / relative
        original = destination.read_bytes()
        destination.unlink()
        with expect_error(AssertionError, "token3 QSA selection"):
            _assert_token3_qsa_selection_blackhole_extension_closure(tmp_path)
        destination.write_bytes(original)

    mismatch = tmp_path / TOKEN3_QSA_SELECTION_BLACKHOLE_EXTENSION_CLOSURE[-1]
    mismatch.write_bytes(mismatch.read_bytes() + b"\n")
    with expect_error(AssertionError, TOKEN3_QSA_SELECTION_BLACKHOLE_EXTENSION_CLOSURE[-1]):
        _assert_token3_qsa_selection_blackhole_extension_closure(tmp_path)


def test_moe_compute_readyseed_relative_header_closure_is_installed_without_flattening() -> None:
    cmake = EXPERIMENTAL_CCL_CMAKE.read_text(encoding="utf-8")
    compute = MOE_COMPUTE_KERNEL.read_text(encoding="utf-8")
    ring = MOE_RING_HEADER.read_text(encoding="utf-8")

    assert cmake.count("moe_compute/device/hostdevcommon/*.hpp") == 1
    # Current upstream installs this relative include through the separate MoEGPT target.
    moe_gpt_cmake = (EXPERIMENTAL_CCL_CMAKE.parent / "moe_gpt/CMakeLists.txt").read_text()
    assert "device/kernels/*.h" in moe_gpt_cmake
    assert "${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt" in moe_gpt_cmake
    assert compute.count('#include "moe_ring_common.h"') == 1
    assert compute.count('#include "../../../moe_gpt/device/kernels/swiglu_sfpu.h"') == 1
    assert ring.count('#include "../hostdevcommon/config.hpp"') == 1
    assert MOE_CONFIG_HEADER.is_file()
    assert MOE_SWIGLU_HEADER.is_file()
