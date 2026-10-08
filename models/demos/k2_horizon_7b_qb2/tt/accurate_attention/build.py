# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build the model-local host binding and private flash-decode kernel adaptation.

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


def prepare_decode_kernel():
    """Stock split-K flash decode with accurate exp and FP32 SFPU recurrence.

    Reader, writer, work split and tree schedule stay stock. The binding switches
    the recurrence CBs to FP32/UnpackToDestFp32; this kernel routes every read of
    them through accurate_decode.hpp. Guarded substitutions fail if upstream changes.
    """
    kernel = (DECODE / "compute/sdpa_flash_decode.cpp").read_text()
    kernel = replace_checked(
        kernel,
        '#include "ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa_block_ops.hpp"\n',
        '#include "ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa_block_ops.hpp"\n'
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
    """Generated decode kernel folder. Prefill attention runs SDPA's ACCURATE precision recipe (see __init__.py)."""
    decode = prepare_decode_kernel()
    fingerprint = hashlib.sha256((decode + (HERE / "accurate_decode.hpp").read_text()).encode()).hexdigest()[:16]
    folder = BUILD / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sdpa_flash_decode.cpp").write_text(decode)
    (folder / "accurate_decode.hpp").write_text((HERE / "accurate_decode.hpp").read_text())
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
    kernel = folder / "sdpa_flash_decode.cpp"
    (BUILD / "kernel_path.txt").write_text(str(kernel) + "\n")
    sources = [
        HERE / "accurate_decode.hpp",
        DECODE / "compute/sdpa_flash_decode.cpp",
        HERE / "binding.cpp",
        HERE / "build.py",
        HERE / "__init__.py",
        ROOT / "build/lib/_ttnncpp.so",
        ROOT / "build/lib/libtt_metal.so",
    ]
    sources += sorted(CORE.glob("*.hpp"))
    generated = [folder / name for name in ("sdpa_flash_decode.cpp", "accurate_decode.hpp")]
    sources += generated
    provenance = {
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "library_sha256": sha256(library),
        "kernel_path": str(kernel),
        "compile_command": command,
        "compiler": subprocess.check_output([command[0], "--version"], text=True).splitlines()[0],
        "generated_files": {str(path.relative_to(ROOT)): sha256(path) for path in generated},
    }
    (BUILD / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(library)
    print(kernel)


if __name__ == "__main__":
    build()
