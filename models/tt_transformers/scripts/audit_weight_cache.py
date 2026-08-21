# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read-only audit of a ttnn weight-cache tree for placeholder-poisoned tensorbins (#45400).

Under the pre-hardening warm-cache gate (branch runs 2026-07-21..2026-08-12), ttnn.as_tensor
could persist torch.empty-derived placeholders as real cache entries for files the seeding
build never wrote (prefetcher ring splits, dtype-variant FF weights). Cold loads cache-HIT such
files forever, so they must be found and deleted, not regenerated over.

For every *.tensorbin under the given roots, prints TSV:
  status  zero_frac  absmax  size_MB  mtime_utc  path
status: POISON-ZERO (>=99.9% zeros), SUSPECT-ERA (variant-pattern name, mtime in the warm-skip
era), LOAD-FAIL (unreadable), OK. Exit code stays 0 -- this is a report, not a gate.
"""
import os
import re
import sys
from datetime import datetime, timezone

import torch
import ttnn

ERA = (datetime(2026, 7, 21, tzinfo=timezone.utc), datetime(2026, 8, 13, tzinfo=timezone.utc))
VARIANT_PATTERNS = re.compile(r"ring|split_sizes|prefetcher|sharded_2d|_bfp4|_bf4|wqkv_bias")


def audit_file(path):
    st = os.stat(path)
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
    try:
        t = ttnn.load_tensor(path)  # host tensor
        try:
            tt = ttnn.to_torch(t)
        except Exception:
            # Mesh-sharded host tensor: stats over the first shard are enough for a zero-scan.
            tt = ttnn.to_torch(ttnn.get_device_tensors(t)[0])
        del t
        flat = tt.flatten().to(torch.float32)
        n = flat.numel()
        zero_frac = float((flat == 0).sum().item()) / max(n, 1)
        absmax = float(flat.abs().max().item()) if n else 0.0
        del tt, flat
    except Exception as e:
        return ("LOAD-FAIL", f"-\t-\t{st.st_size/1e6:.1f}\t{mtime:%Y-%m-%dT%H:%MZ}\t{path}\t# {type(e).__name__}: {e}")
    status = "OK"
    if zero_frac >= 0.999:
        status = "POISON-ZERO"
    elif VARIANT_PATTERNS.search(os.path.basename(path)) and ERA[0] <= mtime <= ERA[1]:
        status = "SUSPECT-ERA"
    return (status, f"{zero_frac:.4f}\t{absmax:.3e}\t{st.st_size/1e6:.1f}\t{mtime:%Y-%m-%dT%H:%MZ}\t{path}")


def main(roots):
    counts = {}
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in sorted(filenames):
                if not fn.endswith(".tensorbin"):
                    continue
                status, line = audit_file(os.path.join(dirpath, fn))
                counts[status] = counts.get(status, 0) + 1
                print(f"[CACHE-AUDIT] {status}\t{line}", flush=True)
    print(f"[CACHE-AUDIT-SUMMARY] {counts}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["/mnt/MLPerf/huggingface/tt_cache/meta-llama/Llama-3.1-8B-Instruct"])
