# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build the model-local host binding and private SDPA kernel adaptation.

This script performs only host compilation; it never imports TTNN or opens a
device. The large shared kernel body is generated from this checkout with
guarded substitutions, so no copied upstream implementation is maintained.
"""

import hashlib
import json
import re
import shlex
import subprocess
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BUILD = HERE / ".build"
CORE = ROOT / "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute"
DATAFLOW = CORE.parent / "dataflow"
DECODE = ROOT / "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels"


def sha256(path):
    # Chunked read: hashlib.file_digest needs Python 3.11; serving images run 3.10.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_checked(text, before, after, count):
    found = text.count(before)
    if found != count:
        raise RuntimeError(f"SDPA source changed: expected {count} occurrences, found {found}: {before!r}")
    return text.replace(before, after)


def prepare_packed_writer():
    """Only change the causal boundary tile; retain reader/compute arithmetic."""
    writer = (DATAFLOW / "writer_interleaved.cpp").read_text()
    # A generated writer is outside the original include directory.
    for name in ("dataflow_common.hpp", "windowed_mask_gen.hpp"):
        writer = replace_checked(writer, f'#include "{name}"', f'#include "{DATAFLOW / name}"', 1)
    start = "    // Lightweight mask: generate template tiles once, leave permanently fronted.\n"
    end = "    // Windowed: load cu_window_seqlens into L1 once;"
    if writer.count(start) != 1 or writer.count(end) != 1:
        raise RuntimeError("Pinned writer mask-generation markers changed")
    original_mask = writer[writer.index(start) : writer.index(end)]
    writer = replace_checked(writer, original_mask, "", 1)
    writer = replace_checked(
        writer,
        "    if constexpr (is_chunked) {\n        if (use_chunk_start_idx_tensor != 0) {",
        "    uint32_t k2_raw_position = 0;\n"
        "    if constexpr (is_chunked) {\n        if (use_chunk_start_idx_tensor != 0) {",
        1,
    )
    writer = replace_checked(
        writer,
        "            uint32_t chunk_start_idx = chunk_start_ptr[0];",
        "            uint32_t chunk_start_idx = chunk_start_ptr[0];\n" "            k2_raw_position = chunk_start_idx;",
        1,
    )
    mask = """    // Packed GQA query rows are distinct heads at the SAME token position.
    // Reader and compute floor raw_position/Q32 exactly as before. Replace the
    // triangular within-tile mask with a vertical cutoff, after receiving the
    // existing writer offset CB. No additional messages or position loads.
    static_assert(NQH == 2 && valid_Sqt == 1 && Sq_chunk_t == 1 && q_num_chunks == 1 && Sk_chunk_t == 4);
    static_assert(is_causal && is_chunked && use_lightweight_mask && !use_streaming_compute);
    static_assert(!use_provided_mask && sliding_window_size == 0 && !use_windowed_mask);
    ASSERT(num_phases == 1 && use_chunk_start_idx_tensor != 0);
    constexpr uint32_t k2_mask_tile_bytes = get_tile_size(cb_mask_in);
    static_assert(k2_mask_tile_bytes == 2048);  // Existing BF16 lightweight masks.
    CircularBuffer k2_masks(cb_mask_in);
    k2_masks.reserve_back(2);
    fill_neginf_tile<k2_mask_tile_bytes>(cb_mask_in, 0);
    fill_vertical_tile_bf16<k2_mask_tile_bytes>(noc, cb_mask_in, 1, (k2_raw_position & 31u) + 1u);
    k2_masks.push_back(2);

"""
    return replace_checked(
        writer, "    uint32_t chunk_start_t_in_q_chunks = 0;", mask + "    uint32_t chunk_start_t_in_q_chunks = 0;", 1
    )


def prepare_decode_kernel():
    """Stock split-K flash decode with accurate exp and FP32 SFPU recurrence.

    Reader, writer, work split and tree schedule stay stock. The binding switches
    the recurrence CBs to FP32/UnpackToDestFp32; this kernel routes every read of
    them through accurate_decode.hpp. Guarded substitutions fail if upstream changes.
    """
    kernel = (DECODE / "compute/sdpa_flash_decode.cpp").read_text()
    kernel = replace_checked(
        kernel,
        '#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"\n',
        '#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"\n'
        '#include "accurate_decode.hpp"\n',
        1,
    )
    kernel = replace_checked(
        kernel,
        "    constexpr bool untilize_output = tilize_q;\n",
        "    constexpr bool untilize_output = tilize_q;\n"
        "    // FP32 recurrence CBs are read only through A2D copies (no untilize/sink/mask paths).\n"
        "    // Full tiles (binding) give VectorMode::RC, hence the accurate clamped exponential;\n"
        "    // stock half-tile decode selects VectorMode::R and its approximate exponential.\n"
        "    static_assert(is_causal && !tilize_q && !use_attention_sink && !use_attention_mask && !use_half_tile);\n",
        1,
    )
    start = "                    /* PREV_SUM *= EXP_MAX_DIFF */\n"
    end = "add_block_inplace<true>(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);\n"
    if kernel.count(start) != 1 or kernel.count(end) != 1:
        raise RuntimeError("Pinned flash-decode recurrence markers changed")
    block = kernel[kernel.index(start) : kernel.index(end) + len(end)]
    for statement in (
        "mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);",
        "cb_out_accumulate_im, cb_exp_max_diff, cb_out_accumulate_im);",
        "add_block_inplace<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);",
    ):
        if block.count(statement) != 1:
            raise RuntimeError("Pinned flash-decode recurrence block changed: " + statement)
    kernel = replace_checked(
        kernel,
        block,
        "                    /* CUR_SUM += PREV_SUM * EXP_MAX_DIFF (FP32 SFPU) */\n"
        "                    k2d_sum_update(cb_cur_sum, cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);\n"
        "                    /* OUT_ACC = OUT_ACC * EXP_MAX_DIFF + OUT_IM (FP32 SFPU) */\n"
        "                    k2d_out_update<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_out_im, cb_exp_max_diff);\n",
        1,
    )
    kernel = replace_checked(
        kernel,
        "reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(\n"
        "                    cb_cur_sum, cb_cur_sum, Sk_chunk_t_dynamic, false);\n",
        "k2d_row_sum<Sq_chunk_t>(cb_qk_im, tt::CBIndex::c_11, cb_cur_sum, Sk_chunk_t_dynamic);\n",
        1,
    )
    kernel = replace_checked(
        kernel, "correction_block<scale_fp32, vector_mode>(", "k2d_correction_block<scale_fp32, vector_mode>(", 1
    )
    kernel = replace_checked(
        kernel,
        "                    mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_exp_max_diff);\n"
        "                    mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im_2, cb_exp_max_diff_2);\n"
        "\n"
        "                    // OUT_ACC = OUT_ACC + OUT_ACC_2\n"
        "                    add_block_inplace<true>(cb_out_accumulate_im, cb_out_accumulate_im_2, out_chunk_tiles);\n",
        "                    k2d_out_merge<Sq_chunk_t, vDHt>(\n"
        "                        cb_out_accumulate_im, cb_exp_max_diff, cb_out_accumulate_im_2, cb_exp_max_diff_2);\n",
        1,
    )
    kernel = replace_checked(
        kernel,
        "            mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_prev_sum);\n",
        "            k2d_scale_output<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_prev_sum);\n",
        1,
    )
    # Upstream routes the once-per-head moves through a non-cloneable move_block_ool wrapper
    # (code size); its body is one of the three remaining direct calls, so rewriting those
    # sends every move, direct or wrapped, through k2d_move.
    kernel = replace_checked(kernel, "move_block_ool(", "move_block_ool(", 9)
    return replace_checked(kernel, "move_block<true>(", "k2d_move<true>(", 3)


def prepare_kernel():
    common = (CORE / "compute_common.hpp").read_text()
    common = replace_checked(common, "enum SDPAType {", '#include "accurate_stats.hpp"\n\nenum SDPAType {', 1)
    common = replace_checked(
        common,
        "mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);",
        "k2_mul_sum_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);",
        2,
    )
    common = replace_checked(
        common,
        "add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);",
        "k2_add_sum_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);",
        2,
    )
    common = replace_checked(
        common,
        "mul_block_bcast_cols<Sq_chunk_t, vDHt, false, true>(",
        "k2_accumulate_output<Sq_chunk_t, vDHt>(",
        1,
    )
    common = replace_checked(
        common,
        "mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(alias_mm2_prev_out, alias_prev_sum, cb_out);",
        "k2_normalize_output<Sq_chunk_t, vDHt>(alias_mm2_prev_out, alias_prev_sum, cb_out);",
        1,
    )
    # Softmax exponential: the FP32 path runs a separate scale pass plus the full-FP32 guarded
    # exponential. fast_exp.hpp fuses the scale and uses a clamped degree-4 base-2 polynomial
    # (2.9e-6 relative), far below the TF32 truncation of P in the PV matmul.
    common = replace_checked(
        common,
        "ALWI void sdpa_reduce_copy_tile_to_dst_init_short(",
        '#include "fast_exp.hpp"\n\nALWI void sdpa_reduce_copy_tile_to_dst_init_short(',
        1,
    )
    common = replace_checked(
        common,
        "                    binop_with_scalar_tile_init();\n"
        "                    mul_unary_tile(j, scale_fp32);\n"
        "                    exp_tile_init<false, 0x3F800000, InputClamping::ClampToNegative>();\n"
        "                    exp_tile<false, false, InputClamping::ClampToNegative, iterations>(j, vector_mode_exp);\n",
        "                    exp_tile_init<false, 0x3F800000, InputClamping::ClampToNegative>();\n"
        "                    MATH((ckernel::sfpu::k2_exp_init()));\n"
        "                    MATH((SFPU_UNARY_CALL(\n"
        "                        DST_SYNC_MODE, DST_ACCUM_MODE, k2_exp_scaled, (iterations), j, vector_mode_exp,\n"
        "                        k2_log2e_scale_bits(scale_fp32))));\n",
        1,
    )
    # The preceding PV operation now packs FP32. Correction factors stay BF16.
    common = replace_checked(
        common,
        "    sub_init(in0_cb, in1_cb);\n    exp_tile_init<EXP_APPROX_MODE>();",
        "    sub_init(in0_cb, in1_cb);\n    pack_reconfig_data_format(out_cb);\n    exp_tile_init<EXP_APPROX_MODE>();",
        1,
    )
    kernel = (CORE / "sdpa.cpp").read_text()
    # Preserve the production streaming header as an include dependency; this
    # binding always configures fp32_dest_acc_en, so that branch is not invoked.
    kernel = replace_checked(
        kernel, '#include "compute_streaming.hpp"', f'#include "{CORE / "compute_streaming.hpp"}"', 1
    )
    writer = prepare_packed_writer()
    decode = prepare_decode_kernel()
    fingerprint = hashlib.sha256(
        (
            common
            + kernel
            + writer
            + (HERE / "accurate_stats.hpp").read_text()
            + decode
            + (HERE / "accurate_decode.hpp").read_text()
            + (HERE / "fast_exp.hpp").read_text()
        ).encode()
    ).hexdigest()[:16]
    folder = BUILD / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "compute_common.hpp").write_text(common)
    (folder / "sdpa.cpp").write_text(kernel)
    (folder / "accurate_stats.hpp").write_text((HERE / "accurate_stats.hpp").read_text())
    (folder / "writer_packed_gqa.cpp").write_text(writer)
    (folder / "sdpa_flash_decode.cpp").write_text(decode)
    (folder / "accurate_decode.hpp").write_text((HERE / "accurate_decode.hpp").read_text())
    (folder / "fast_exp.hpp").write_text((HERE / "fast_exp.hpp").read_text())
    return folder


def build():
    folder = prepare_kernel()
    # Match the existing TTNN extension's compiler flags, ABI and dependency
    # include paths. A standalone binding does not use the main tree's PCH.
    ninja = (ROOT / "build/build.ninja").read_text()
    match = re.search(r"^build ttnn/CMakeFiles/ttnn\.dir/[^\n]*\n((?:  [^\n]*\n)+)", ninja, re.M)
    if not match:
        raise RuntimeError("Build TTNN first: its extension compile configuration was not found")
    fields = dict(re.findall(r"^  (\w+) = (.*)$", match.group(1), re.M))
    command = ["clang++-20", "-std=c++20", "-shared", "-fPIC", "-O2", "-fvisibility=hidden"]
    command += shlex.split(fields["DEFINES"]) + shlex.split(fields["INCLUDES"])
    command += ["-I" + sysconfig.get_path("include"), str(HERE / "binding.cpp")]
    command += [str(ROOT / "build/ttnn/libnanobind-static-abi3.a")]
    command += [str(ROOT / "build/lib/_ttnncpp.so"), str(ROOT / "build/lib/libtt_metal.so")]
    command += ["-Wl,-rpath," + str(ROOT / "build/lib")]
    library = BUILD / ("_k2_accurate_attention" + sysconfig.get_config_var("EXT_SUFFIX"))
    temporary_library = library.with_suffix(library.suffix + ".tmp")
    command += ["-o", str(temporary_library)]
    subprocess.run(command, check=True, cwd=ROOT)
    temporary_library.replace(library)
    (BUILD / "kernel_path.txt").write_text(str(folder / "sdpa.cpp") + "\n")
    sources = [
        CORE / "compute_common.hpp",
        CORE / "sdpa.cpp",
        HERE / "accurate_stats.hpp",
        HERE / "accurate_decode.hpp",
        HERE / "fast_exp.hpp",
        DECODE / "compute/sdpa_flash_decode.cpp",
        HERE / "binding.cpp",
        HERE / "build.py",
        HERE / "__init__.py",
        ROOT / "build/lib/_ttnncpp.so",
        ROOT / "build/lib/libtt_metal.so",
        DATAFLOW / "reader_interleaved.cpp",
        DATAFLOW / "writer_interleaved.cpp",
    ]
    sources += sorted(CORE.glob("*.hpp")) + sorted(DATAFLOW.glob("*.hpp"))
    generated = [
        folder / name
        for name in (
            "compute_common.hpp",
            "sdpa.cpp",
            "accurate_stats.hpp",
            "writer_packed_gqa.cpp",
            "sdpa_flash_decode.cpp",
            "accurate_decode.hpp",
            "fast_exp.hpp",
        )
    ]
    sources += generated
    provenance = {
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "library_sha256": sha256(library),
        "kernel_path": str(folder / "sdpa.cpp"),
        "compile_command": command,
        "compiler": subprocess.check_output([command[0], "--version"], text=True).splitlines()[0],
        "generated_files": {str(path.relative_to(ROOT)): sha256(path) for path in generated},
        "packed_gqa_mask": {
            "contract": "same_raw_position_for_all_query_rows_v1",
            "writer_path": str(folder / "writer_packed_gqa.cpp"),
            "writer_sha256": sha256(folder / "writer_packed_gqa.cpp"),
            "mask_semantics_sha256": hashlib.sha256(
                (
                    (folder / "writer_packed_gqa.cpp").read_text() + (DATAFLOW / "dataflow_common.hpp").read_text()
                ).encode()
            ).hexdigest(),
            "source_dependencies": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (
                    DATAFLOW / "writer_interleaved.cpp",
                    DATAFLOW / "dataflow_common.hpp",
                    DATAFLOW / "windowed_mask_gen.hpp",
                )
            },
            "required_position_residues": list(range(32)),
        },
    }
    (BUILD / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(library)
    print(folder / "sdpa.cpp")


if __name__ == "__main__":
    build()
