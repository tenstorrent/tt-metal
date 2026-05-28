"""Test what Pyright says about real untyped method-call receivers."""
import sys
sys.path.insert(0, '/workspace/dep-graph/scripts')
from pathlib import Path
from pyright_query import PyrightClient
import json

# Pick 10 untyped method calls from inventory and query pyright
py = json.load(open('/workspace/dep-graph/cache/py_index.json'))
SKIP = {"ttnn", "self", "cls", "torch", "numpy", "os", "sys", "math", "json", "pytest",
        "logger", "argparse", "transformers"}

# Find diverse samples
samples = []
seen_pairs = set()
for r in py['refs']:
    ch = r['target_chain']
    if len(ch) < 2: continue
    if ch[0] in SKIP: continue
    if r.get('receiver_type'): continue
    pair = (ch[0], ch[1])
    if pair in seen_pairs: continue
    seen_pairs.add(pair)
    samples.append(r)
    if len(samples) >= 15:
        break

client = PyrightClient(Path('/workspace'))
client.start()

try:
    for r in samples:
        ch = r['target_chain']
        site_file = r['site_file']
        line = r['site_line'] - 1  # pyright uses 0-indexed lines
        col = r.get('site_col', 0)
        # Read the line to find receiver position
        text = Path(site_file).read_text().split('\n')
        if line >= len(text):
            continue
        src_line = text[line]
        # Find chain[0] receiver position
        recv = ch[0]
        idx = src_line.find(recv)
        if idx < 0:
            continue
        client.open_file(Path(site_file))
        result = client.hover(Path(site_file), line, idx)
        print(f"chain={ch[:3]}")
        print(f"  site: {site_file.split('/')[-1]}:{line+1}")
        print(f"  pyright: {result.inferred_type!r}")
        print()
finally:
    client.shutdown()
