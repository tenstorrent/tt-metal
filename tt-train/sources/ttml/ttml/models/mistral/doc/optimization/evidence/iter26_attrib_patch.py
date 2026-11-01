"""Stage-removal patches for the SDPA compute kernels (attribution only; the outputs are wrong).
usage: iter26_attrib_patch.py <kernel: fw|kv|q> <variant>[,<variant>...]   (applied in place; restore with git)"""
import re
import sys

ROOT = "/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/metal/ops"
FILES = {
    "fw": [
        f"{ROOT}/sdpa_fw/device/kernels/compute/sdpa_fw_compute_kernel.cpp",
        f"{ROOT}/sdpa_fw/device/kernels/compute/sdpa_compute_utils.hpp",
    ],
    "kv": [
        f"{ROOT}/sdpa_bw/device/kernels/compute/sdpa_bw_kv_compute_kernel.cpp",
        f"{ROOT}/sdpa_bw/device/kernels/compute/sdpa_bw_compute_utils.hpp",
    ],
    "q": [
        f"{ROOT}/sdpa_bw/device/kernels/compute/sdpa_bw_q_compute_kernel.cpp",
        f"{ROOT}/sdpa_bw/device/kernels/compute/sdpa_bw_compute_utils.hpp",
    ],
    "fwdf": [
        f"{ROOT}/sdpa_fw/device/kernels/dataflow/sdpa_fw_reader_kernel.cpp",
        f"{ROOT}/sdpa_fw/device/kernels/dataflow/sdpa_fw_writer_kernel.cpp",
    ],
}
# variant -> list of (regex, replacement, min occurrences)
P = {
    "fw": {
        "noqk": [(r"matmul_block\(\s*cb_query,\s*cb_key,[^;]*;", ";", 1)],
        "nomax": [(r"reduce_tile<pool_type, reduce_dim>\([^;]*;", ";", 1)],
        "nosub": [(r"sub_tiles_bcast_cols\(\s*cb_attention_weights, cb_cur_max,[^;]*;", ";", 1)],
        "noexp": [(r"sdpa_exp_tile_scaled<scaler_fp32, scaler_bf16>\(n\);", ";", 1)],
        "nopv": [(r"matmul_block\(\s*cb_attention_weights,\s*cb_value,[^;]*;", ";", 1)],
        "nocorr": [(r"update_cur_mm_out\(vWt, block_size,[^;]*;", ";", 1)],
        "nomask": [(r"if \(k_chunk == diag_chunk\) \{", "if (false) {", 1)],
        "nopack": [(r"pack_tile(</\* out_of_order \*/ true>)?\([^;]*;", ";", 10)],
        "noacc": [
            (r"pack_tile\(block_idx, cb_cur_mm_out\);", ";", 1),
            (r"pack_tile</\* out_of_order \*/ true>\(/\* dst \*/ n, cb_cur_exp_sum, /\* offset \*/ 0\);", ";", 1),
        ],
        "noregs": [(r"tile_regs_(acquire|commit|wait|release)\(\);", ";", 20)],
        "noreconfig": [(r"\b(reconfig_data_format(_srca|_srcb)?|pack_reconfig_data_format)\([^;]*;", ";", 20)],
        "noinit": [
            (r"init_unary_bcast_col\((alias_)?cb_[a-z_]*\);", ";", 2),
            (
                r"\b(reconfig_data_format(_srca|_srcb)?|matmul_block_init|matmul_init|reduce_init<[^>]*>|reduce_uninit|copy_init|sub_bcast_cols_init|mul_bcast_cols_init|mul_binary_tile_init|add_binary_tile_init|binary_max_tile_init|mask_tile_init|binop_with_scalar_tile_init|recip_tile_init|log_tile_init<[^>]*>|pack_reconfig_data_format|pack_reconfig_l1_acc|sub_init|sdpa_exp_tile_init)\([^;]*;",
                ";",
                20,
            ),
        ],
        "nocopy": [
            (
                r"\b(copy_tile|unary_bcast<BroadcastType::COL>|sub_tiles|mul_tiles_bcast_cols|binary_max_tile|add_binary_tile|mul_binary_tile|mask_tile|mul_unary_tile|add_unary_tile|log_tile<[^>]*>|sdpa_exp_tile_first_column_scaled<[^>]*>|matmul_tiles)\([^;]*;",
                ";",
                10,
            ),
            (r"MATH\(\(recip_tile_first_column\(dst_idx\)\)\);", ";", 1),
        ],
    },
    "kv": {
        "noqk": [(r"matmul_tiles\(\s*cb_query,\s*cb_key,[^;]*;", ";", 1)],
        "noexp": [(r"sdpa_exp_tile\(scores_reg\);", ";", 1)],
        "nodp": [(r"matmul_tiles\(\s*cb_grad_output,\s*cb_value,[^;]*;", ";", 1)],
        "nods": [
            (r"transpose_dest</\* is_32bit \*/ true>\((grad_reg|attn_weights_reg)\);", ";", 2),
            (r"mul_binary_tile\(grad_reg, attn_weights_reg, grad_reg\);", ";", 1),
            (r"copy_tile\(cb_attention_weights, /\* tile_idx \*/ 0, /\* register idx \*/ attn_weights_reg\);", ";", 1),
            (r"sub_tiles_bcast_cols\(\s*cb_grad_attn_weights,\s*cb_u_scalar_row,[^;]*;", ";", 1),
        ],
        "nodv": [(r"matmul_tiles\(\s*cb_attn_weights_transposed,\s*cb_grad_output,[^;]*;", ";", 1)],
        "nodk": [(r"matmul_tiles\(\s*cb_grad_scores_transposed,\s*cb_query,[^;]*;", ";", 1)],
        "nopack": [(r"pack_tile(</\* out_of_order \*/ true>)?\([^;]*;", ";", 10)],
        "noacc": [(r"pack_tile\(/\*dst_reg_idx\*/ block_idx, cb_grad_(value|key)_accum\);", ";", 2)],
    },
    "fwdf": {
        "nofwd": [(r"noc\.async_write\(\s*CoreLocalMem<uint32_t>\([kv]_addr\),[^;]*;", ";", 2)],
        "noread": [
            (r"read_tile_block_transposed</\* UseBarrier \*/ false>\([^;]*;", ";", 1),
            (r"read_tiles_by_row</\* UseBarrier \*/ false>\(\s*cb_value,[^;]*;", ";", 1),
        ],
    },
    "q": {
        "noqk": [(r"matmul_tiles\(cb_query, cb_key, /\* in0 \*/ f,[^;]*;", ";", 1)],
        "noexp": [(r"sdpa_exp_tile\(first_reg \+ i\);", ";", 1)],
        "nodp": [(r"matmul_tiles\(\s*cb_dp_grad_output, cb_value,[^;]*;", ";", 1)],
        "nods": [
            (r"sub_tiles_bcast_cols\(cb_grad_attn_weights, cb_u_scalar_row, /\* tile_idx \*/ c0 \+ j,[^;]*;", ";", 1),
            (r"copy_tile\(cb_attention_weights, /\* tile_idx \*/ c0 \+ j,[^;]*;", ";", 1),
            (r"mul_binary_tile\(j, 2U \+ j, j\);", ";", 1),
        ],
        "nodq": [(r"matmul_tiles\(cb_grad_scores, cb_key, /\* in0 \*/ j,[^;]*;", ";", 1)],
        "nopack": [(r"pack_tile(</\* out_of_order \*/ true>)?\([^;]*;", ";", 10)],
        "noacc": [(r"pack_tile\(b, cb_grad_query_accum\);", ";", 1)],
    },
}
kernel, variants = sys.argv[1], sys.argv[2].split(",")
if variants == ["base"]:
    print("base: no patch")
    sys.exit(0)
if variants[0] == "skeleton":
    extra = variants[1:]
    variants = [
        v for v in P[kernel] if v not in ("nomask", "nopack", "noacc", "noregs", "noinit", "nocopy", "noreconfig")
    ] + extra
texts = {f: open(f).read() for f in FILES[kernel]}
for v in variants:
    for rx, rep, nmin in P[kernel][v]:
        total = 0
        for f in FILES[kernel]:
            new, n = re.subn(rx, rep, texts[f], flags=re.S)
            texts[f] = new
            total += n
        if total < nmin:
            print(f"PATCH FAILED: {kernel} {v} pattern {rx!r} matched {total} (< {nmin})")
            sys.exit(2)
        print(f"{kernel} {v}: {rx[:50]!r} x{total}")
for f, t in texts.items():
    open(f, "w").write(t)
print("patched", kernel, variants)
