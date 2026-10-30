# SPDX-License-Identifier: Apache-2.0
"""Local evidence receipts; raw profiles/logs stay local, never in telemetry."""
import hashlib
import json
from pathlib import Path

out=Path(__file__).resolve().parents[1]/'doc/optimized_full_model'
# Normalize generated table formatting only; raw device/op CSVs remain untouched.
for path in [out/'head_candidates.csv',*out.glob('tracy/b*/*_perf_report.csv')]:
    if path.exists():path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
for path in out.glob('tracy/b*/*_perf_report.txt'):
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()).rstrip()+'\n')
rows=[]
for path in sorted(out.rglob('*')):
    if not path.is_file() or path.name=='artifact_manifest.json' or path.suffix in ('.md','.pyc'):
        continue
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    rows.append(dict(path=str(path.relative_to(out)),bytes=path.stat().st_size,sha256=digest.hexdigest()))
(out/'artifact_manifest.json').write_text(json.dumps(dict(scope='Local runtime evidence receipts. Markdown reports excluded because checkpoint/review receipts update them; raw logs/CSV/Tracy remain local. No raw evidence is copied into telemetry.',files=rows),indent=2)+'\n')
print(len(rows),'files',sum(r['bytes'] for r in rows),'bytes')
