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
    fingerprint = hashlib.sha256(
        (common + kernel + (HERE / "accurate_stats.hpp").read_text() + (HERE / "fast_exp.hpp").read_text()).encode()
    ).hexdigest()[:16]
    folder = BUILD / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "compute_common.hpp").write_text(common)
    (folder / "sdpa.cpp").write_text(kernel)
    (folder / "accurate_stats.hpp").write_text((HERE / "accurate_stats.hpp").read_text())
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
        HERE / "fast_exp.hpp",
        HERE / "binding.cpp",
        HERE / "build.py",
        HERE / "__init__.py",
        ROOT / "build/lib/_ttnncpp.so",
        ROOT / "build/lib/libtt_metal.so",
        DATAFLOW / "reader_interleaved.cpp",
        DATAFLOW / "writer_interleaved.cpp",
    ]
    sources += sorted(CORE.glob("*.hpp")) + sorted(DATAFLOW.glob("*.hpp"))
    generated = [folder / name for name in ("compute_common.hpp", "sdpa.cpp", "accurate_stats.hpp", "fast_exp.hpp")]
    sources += generated
    provenance = {
        "files": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "library_sha256": sha256(library),
        "kernel_path": str(folder / "sdpa.cpp"),
        "compile_command": command,
        "compiler": subprocess.check_output([command[0], "--version"], text=True).splitlines()[0],
        "generated_files": {str(path.relative_to(ROOT)): sha256(path) for path in generated},
    }
    (BUILD / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(library)
    print(folder / "sdpa.cpp")


if __name__ == "__main__":
    build()
