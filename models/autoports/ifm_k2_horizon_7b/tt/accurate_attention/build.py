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
    fingerprint = hashlib.sha256(
        (common + kernel + writer + (HERE / "accurate_stats.hpp").read_text()).encode()
    ).hexdigest()[:16]
    folder = BUILD / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "compute_common.hpp").write_text(common)
    (folder / "sdpa.cpp").write_text(kernel)
    (folder / "accurate_stats.hpp").write_text((HERE / "accurate_stats.hpp").read_text())
    (folder / "writer_packed_gqa.cpp").write_text(writer)
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
        folder / name for name in ("compute_common.hpp", "sdpa.cpp", "accurate_stats.hpp", "writer_packed_gqa.cpp")
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
