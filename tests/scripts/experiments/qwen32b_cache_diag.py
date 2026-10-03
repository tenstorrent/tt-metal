# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic for #58871: inspect the shared Qwen3-32B Galaxy tensor cache.

Prints, for each cache sub-directory under <TT_CACHE_PATH>/TG, the file mtime histogram and,
for a sample of the DRAM-sharded (prefetcher) weight files, the memory config stored in the
flatbuffer. A shard grid of 8 DRAM cores means the file was written on Blackhole; 12 means Wormhole.
"""
import glob
import os
import socket
import time
from collections import Counter

import ttnn

root = os.path.join(os.environ.get("TT_CACHE_PATH", "/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen3-32B"), "TG")
print(f"host={socket.gethostname()} arch={ttnn.get_arch_name()} root={root}", flush=True)

PATTERNS = (
    "*layers.0.*prefetcher*",
    "*layers.0.*sharded_2d*",
    "*layers.0.*dram*",
    "*lm_head*",
    "*output*",
    "*pb_rs*",
)

for sub in ("tensor_cache_bf16", "tensor_cache_bfp8", "tensor_cache_instruct_bf16", "tensor_cache_instruct_bfp8"):
    d = os.path.join(root, sub)
    if not os.path.isdir(d):
        print(f"== {sub}: missing", flush=True)
        continue
    files = sorted(glob.glob(os.path.join(d, "*.tensorbin")))
    days = Counter(time.strftime("%Y-%m-%d", time.gmtime(os.stat(f).st_mtime)) for f in files)
    print(f"== {sub}: {len(files)} files; mtime days: {dict(sorted(days.items()))}", flush=True)
    seen = set()
    for pat in PATTERNS:
        for f in sorted(glob.glob(os.path.join(d, pat)))[:4]:
            if f in seen:
                continue
            seen.add(f)
            st = os.stat(f)
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(st.st_mtime))
            try:
                t = ttnn.load_tensor(f)
                mc = t.memory_config()
                ss = getattr(mc, "shard_spec", None)
                grid = ss.grid if ss is not None else None
                ncores = grid.num_cores() if grid is not None else 0
                desc = (
                    f"shape={tuple(t.shape)} layout={mc.memory_layout} buffer={mc.buffer_type} "
                    f"shard_cores={ncores} shard_shape={ss.shape if ss is not None else None} grid={grid}"
                )
            except Exception as e:  # noqa: BLE001 - diagnostic only
                desc = f"LOAD FAILED: {str(e)[:300]}"
            print(f"{mtime} {st.st_size:>12} {os.path.basename(f)} {desc}", flush=True)
