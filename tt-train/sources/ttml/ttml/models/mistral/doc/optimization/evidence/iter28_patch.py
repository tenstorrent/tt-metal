"""Iteration 28: the KV backward kernel processes query rows in blocks (the chain chunk): scores two per DEST pass,
dP / dS in one pass per row (dP never leaves DEST), dV and dK contracted over the block in DEST with one L1
accumulate per block. Installs the kernel from the scratchpad copy and sizes the block CBs / adds the q_block arg."""
import shutil
import sys

M = "tt-train/sources/ttml/metal/ops/sdpa_bw/device"
KF = f"{M}/sdpa_bw_kv_program_factory.cpp"
KC = f"{M}/kernels/compute/sdpa_bw_kv_compute_kernel.cpp"
SP = "/tmp/claude-1000/-home-ttuser-agobeljic-tt-metal/0cbc71c0-30d2-4e45-a0c0-f0a2063e85a7/scratchpad"


def sub(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        print(f"FAILED {path}: {old[:90]!r} found {n}, expected {count}")
        sys.exit(2)
    open(path, "w").write(s.replace(old, new))


shutil.copy(f"{SP}/iter28_sdpa_bw_kv_compute_kernel.cpp", KC)
sub(
    KF,
    "    const uint32_t block_size_q = get_block_size(qWt, 4U);\n    const uint32_t block_size_v = get_block_size(vWt, 4U);\n",
    "    const uint32_t block_size_q = get_block_size(qWt, 4U);\n    const uint32_t block_size_v = get_block_size(vWt, 4U);\n"
    "    // Query rows per compute block (iteration 28): the chain chunk under CHAIN (whole chunks resident), 1 otherwise.\n"
    "    const uint32_t q_block = use_chain ? chain_chunk : 1U;\n",
)
for cb in ("kAttnWeightsTransposedCbIndex", "kAttentionWeightsCbIndex", "kGradScoresTransposedCbIndex"):
    sub(
        KF,
        f"        {cb},\n        precise_data_format,\n        float32_single_tile_size_bytes,\n        kSingleTileBuffer);\n",
        f"        {cb},\n        precise_data_format,\n        float32_single_tile_size_bytes,\n        q_block);  // one block (iteration 28)\n",
    )
s = open(KF).read()
n = s.count("custom_inf           // 9: used to transform mask from 0/-1 to 0/-inf\n")
n2 = s.count("custom_inf                  // 9: used to transform mask from 0/-1 to 0/-inf\n")
print("ct arg lists:", n, n2)
s = s.replace(
    "custom_inf           // 9: used to transform mask from 0/-1 to 0/-inf\n",
    "custom_inf,          // 9: used to transform mask from 0/-1 to 0/-inf\n            q_block              // 10: query rows per compute block\n",
)
s = s.replace(
    "custom_inf                  // 9: used to transform mask from 0/-1 to 0/-inf\n",
    "custom_inf,                 // 9: used to transform mask from 0/-1 to 0/-inf\n            q_block                     // 10: query rows per compute block\n",
)
open(KF, "w").write(s)
print("iteration 28 patch applied")
