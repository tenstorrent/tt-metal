# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only exhaustive SDPA scheduling bound; never imports TTNN."""

import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "tt/multichip_decoder.py"
tree = ast.parse(source.read_text())
function = next(
    node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "sdpa_decode_key_chunk"
)
namespace = {}
exec(  # noqa: S102 - trusted local function, no TTNN import
    compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace
)
select = namespace["sdpa_decode_key_chunk"]
checks = 0
transitions = []
for batch in range(1, 33):
    workers = min(16, 110 // batch)
    previous = None
    for pages in range(1, 8193):
        chunk = select(batch, pages, 110)
        # Independent inequalities cover every legal tile-aligned capacity,
        # not just the measured 4K and 128K points.
        assert chunk in (64, 128, 256, 512)
        assert pages * 32 <= workers * chunk * 200
        assert chunk == 64 or pages * 32 > workers * (chunk // 2) * 200
        alignment = max(4, chunk // 32)
        padded_pages = ((pages + alignment - 1) // alignment) * alignment
        assert padded_pages >= pages and padded_pages % (chunk // 32) == 0
        if chunk != previous:
            transitions.append({"batch": batch, "first_pages": pages, "key_chunk": chunk, "workers": workers})
            previous = chunk
        checks += 1
for pages in range(1, 8193):
    assert select(1, pages, 64) == select(1, pages, 110)
    checks += 1
result = {
    "status": "passed",
    "checks": checks,
    "max_context": 262144,
    "max_local_chunk_updates": 200,
    "transitions": transitions,
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
}
(root / "doc/optimized_multichip_decoder/final_sdpa_policy_host.json").write_text(json.dumps(result, indent=2) + "\n")
print("SDPA_POLICY_HOST_PASS", checks)
